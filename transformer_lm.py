# transformer_lm.py

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from utils import *

class LanguageModel(object):
    def get_next_char_log_probs(self, context):
        raise Exception("Only implemented in subclasses")

    def get_log_prob_sequence(self, next_chars, context):
        raise Exception("Only implemented in subclasses")
    
class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0 / self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0 / self.voc_size) * len(next_chars)

class NeuralLMModule(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=8, num_layers=3, dim_feedforward=512, dropout=0.1, max_len=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=False)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        self.output.weight = self.embedding.weight
        self.log_softmax = nn.LogSoftmax(dim=-1)
        pe = self._build_pos_encoding(max_len, d_model)
        self.register_buffer("pos_embedding", pe)


# switched from learned positional embeddings to sinusoidal positional encodings for stability and generalization
    def _build_pos_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  

    def forward(self, indices):
        device = next(self.parameters()).device
        indices = indices.to(device)
        seq_len = indices.size(0)
        x = self.embedding(indices) * math.sqrt(self.d_model)  
        pos = self.pos_embedding[:seq_len, :].to(device)
        x = x + pos  
        x = x.unsqueeze(1)  
        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device), 1)
        x = self.transformer(x, mask=mask)  
        x = x.squeeze(1)  
        logits = self.output(x)  
        logits = torch.nan_to_num(logits, neginf=-1e9)
        log_probs = self.log_softmax(logits)  
        return log_probs

class NeuralLanguageModel(LanguageModel):
    def __init__(self, pytorch_model: NeuralLMModule, vocab_index: Indexer):
        self.model = pytorch_model
        self.vocab_index = vocab_index
        self.device = next(self.model.parameters()).device
        self.model.eval()

    def get_next_char_log_probs(self, context:str) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            space_id = self.vocab_index.index_of(' ')
            idxs = [self.vocab_index.index_of(c) for c in (context or "")]
            idxs = [i if i >= 0 else space_id for i in idxs]

            if len(idxs) == 0:
                idxs = [space_id]

            max_pos = int(self.model.pos_embedding.shape[0])
            if len(idxs) > max_pos:
                idxs = idxs[-max_pos:]

            indices = torch.LongTensor(idxs).to(self.device)  
            log_probs = self.model(indices)                 
            return log_probs[-1].cpu().numpy()  
            
    def get_log_prob_sequence(self, next_chars: str, context: str) -> float:
        self.model.eval()
        total = 0.0
        ctx = context or ""
        for ch in (next_chars or ""):
            lp = self.get_next_char_log_probs(ctx)     # [vocab_size]
            idx = self.vocab_index.index_of(ch)
            if idx == -1:
                idx = self.vocab_index.index_of(' ')
            total += float(lp[idx])
            ctx += ch
        return total


# Training 
def train_lm(args, train_text, dev_text, vocab_index):
    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    device = torch.device("cpu")
    vocab_size = len(vocab_index)  
    d_model = 192
    nhead = 8
    num_layers = 4
    dim_feedforward = 768
    dropout = 0.1
    max_len = 1024
    seq_len = 128        
    batch_size = 64     
    num_epochs = 10
    lr = 3e-4

    # Create PyTorch model
    pytorch_model = NeuralLMModule(vocab_size=vocab_size, d_model=d_model, nhead=nhead,
                                   num_layers=num_layers, dim_feedforward=dim_feedforward,
                                   dropout=dropout, max_len=max_len)
    pytorch_model.to(device)

    optimizer = optim.Adam(pytorch_model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)
    loss_fn = nn.NLLLoss()
   
    train_indices = [vocab_index.index_of(c) for c in train_text]
    unk_idx = vocab_index.index_of(' ')
    train_indices = [i if i >= 0 else unk_idx for i in train_indices]
    N = len(train_indices)

    def chunk_generator(indices, seq_len, stride=None):
        if stride is None:
            stride = seq_len // 4  
        i = 0
        while i + seq_len < len(indices):
            chunk = indices[i:i + seq_len + 1]
            if len(chunk) == seq_len + 1:
                yield chunk[:-1], chunk[1:]
            i += stride

    # Training loop
    pytorch_model.train()
    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        count = 0
        for X_chunk, Y_chunk in chunk_generator(train_indices, seq_len, stride=seq_len//2):
            X = torch.LongTensor(X_chunk).to(device)
            Y = torch.LongTensor(Y_chunk).to(device)
            optimizer.zero_grad()
            log_probs = pytorch_model(X)  
            loss = loss_fn(log_probs.view(-1, vocab_size), Y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pytorch_model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            count += 1
        avg_loss = total_loss / max(1, count)
        wrapper_tmp = NeuralLanguageModel(pytorch_model, vocab_index)
        dev_logprob = wrapper_tmp.get_log_prob_sequence(dev_text, "")
        dev_ppl = math.exp(-dev_logprob / max(1, len(dev_text)))
        print(f"Epoch {epoch}/{num_epochs} - train_loss={avg_loss:.4f} - dev_perplexity={dev_ppl:.4f}")
        scheduler.step()
    pytorch_model.eval()

    nm = NeuralLanguageModel(pytorch_model, vocab_index)
    return nm
