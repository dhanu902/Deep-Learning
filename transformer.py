# transformer.py

import time
import torch
import torch.nn as nn
import numpy as np
import random
from torch import optim
import matplotlib.pyplot as plt
from typing import List
from utils import *

class LetterCountingExample(object):
    def __init__(self, input: str, output: np.array, vocab_index: Indexer):
        self.input = input
        self.input_indexed = np.array([vocab_index.index_of(ci) for ci in input])
        self.input_tensor = torch.LongTensor(self.input_indexed)
        self.output = output
        self.output_tensor = torch.LongTensor(self.output)

class Transformer(nn.Module):
    def __init__(self, vocab_size, num_positions, d_model, d_internal, num_classes, num_layers, attend_backward_only: bool = True):

        super().__init__()

        self.embed = nn.Embedding(vocab_size, d_model)
        self.posenc = PositionalEncoding(d_model, num_positions=num_positions, batched=False)
        self.layers = nn.ModuleList([TransformerLayer(d_model, d_internal) for _ in range(num_layers)])
        self.out = nn.Linear(d_model, num_classes)
        self.logsm = nn.LogSoftmax(dim=1)
        self.attend_backward_only = attend_backward_only

    def forward(self, indices):
        # indices: [20]
        x = self.embed(indices)                 
        x = self.posenc(x)                       
        L = x.size(0)

        attn_mask = None
        if self.attend_backward_only:

            attn_mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), diagonal=1)

        attn_maps = []
        for layer in self.layers:
            x, attn = layer(x, attn_mask=attn_mask)                   
            attn_maps.append(attn)
        logits = self.out(x)                     
        log_probs = self.logsm(logits)           
        return log_probs, attn_maps

class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_internal):

        super().__init__()
        self.q = nn.Linear(d_model, d_internal)
        self.k = nn.Linear(d_model, d_internal)
        self.v = nn.Linear(d_model, d_internal)
        self.o = nn.Linear(d_internal, d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, input_vecs, attn_mask=None):
        
        x = self.ln1(input_vecs)
        Q = self.q(x)                            
        K = self.k(x)                             
        V = self.v(x)                            
        scale = (K.shape[-1]) ** 0.5
        scores = (Q @ K.transpose(0, 1)) / scale  

        if attn_mask is not None:
             
            scores = scores.masked_fill(attn_mask, float('-inf'))

        attn = torch.softmax(scores, dim=-1)      
        context = attn @ V                       
        x = input_vecs + self.dropout(self.o(context))    

        y = self.ln2(x)
        y = self.dropout(self.ff(y))                        
        out = x + y                             
        return out, attn
 
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int=20, batched=False):

        super().__init__()

        self.emb = nn.Embedding(num_positions, d_model)
        self.batched = batched

    def forward(self, x):
        input_size = x.shape[-2]
        indices_to_embed = torch.arange(input_size, device=x.device, dtype=torch.long)
        if self.batched:
            emb_unsq = self.emb(indices_to_embed).unsqueeze(0)
            return x + emb_unsq
        else:
            return x + self.emb(indices_to_embed)


def train_classifier(args, train, dev):
    
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    attend_backward_only = True if args.task == "BEFORE" else False
    
    vocab_size = 27
    num_positions = 20
    d_model = 64
    d_internal = 64
    num_classes = 3
    num_layers = 2

    model = Transformer(vocab_size, num_positions, d_model, d_internal, num_classes, num_layers, attend_backward_only=attend_backward_only)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fcn = nn.NLLLoss()

    model.train()
    num_epochs = 10

    best_acc, best_state = -1.0, None

    for epoch in range(num_epochs):
        ex_idxs = list(range(len(train)))
        random.shuffle(ex_idxs)
        epoch_loss = 0.0

        for i in ex_idxs:
            ex = train[i]
            log_probs, _ = model(ex.input_tensor)              
            loss = loss_fcn(log_probs, ex.output_tensor.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs} - loss={epoch_loss:.4f}")

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for ex in dev[:200]:  
                logp, _ = model(ex.input_tensor)
                pred = torch.argmax(logp, dim=1)
                correct += int((pred == ex.output_tensor).sum())
                total += len(pred)
        dev_acc = correct / total if total else 0.0
        print(f"  dev_acc={dev_acc:.4f}")

        if dev_acc > best_acc:
            best_acc = dev_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        model.train()

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    return model


####################################
# DO NOT MODIFY IN YOUR SUBMISSION #
####################################
def decode(model: Transformer, dev_examples: List[LetterCountingExample], do_print=False, do_plot_attn=False):
    """
    Decodes the given dataset, does plotting and printing of examples, and prints the final accuracy.
    :param model: your Transformer that returns log probabilities at each position in the input
    :param dev_examples: the list of LetterCountingExample
    :param do_print: True if you want to print the input/gold/predictions for the examples, false otherwise
    :param do_plot_attn: True if you want to write out plots for each example, false otherwise
    :return:
    """
    num_correct = 0
    num_total = 0
    if len(dev_examples) > 100:
        print("Decoding on a large number of examples (%i); not printing or plotting" % len(dev_examples))
        do_print = False
        do_plot_attn = False
    for i in range(0, len(dev_examples)):
        ex = dev_examples[i]
        (log_probs, attn_maps) = model.forward(ex.input_tensor)
        predictions = np.argmax(log_probs.detach().numpy(), axis=1)
        if do_print:
            print("INPUT %i: %s" % (i, ex.input))
            print("GOLD %i: %s" % (i, repr(ex.output.astype(dtype=int))))
            print("PRED %i: %s" % (i, repr(predictions)))
        if do_plot_attn:
            for j in range(0, len(attn_maps)):
                attn_map = attn_maps[j]
                fig, ax = plt.subplots()
                im = ax.imshow(attn_map.detach().numpy(), cmap='hot', interpolation='nearest')
                ax.set_xticks(np.arange(len(ex.input)), labels=ex.input)
                ax.set_yticks(np.arange(len(ex.input)), labels=ex.input)
                ax.xaxis.tick_top()
                # plt.show()
                plt.savefig("plots/%i_attns%i.png" % (i, j))
        acc = sum([predictions[i] == ex.output[i] for i in range(0, len(predictions))])
        num_correct += acc
        num_total += len(predictions)
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))
