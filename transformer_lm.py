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

class NeuralLMModule(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=8, num_layers=3, dim_feedforward=512, dropout=0.1, max_len=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=False)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Linear(d_model, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        pe = self._build_pos_encoding(max_len, d_model)
        self.register_buffer("pos_embedding", pe)

    def _build_pos_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # register as buffer when used in module; but we return a tensor here and register later if needed
        return pe  # shape [max_len, d_model]

    def forward(self, indices):

        # Expect indices shape [seq_len]
        device = next(self.parameters()).device
        indices = indices.to(device)
        seq_len = indices.size(0)
        x = self.embedding(indices) * math.sqrt(self.d_model)  # [seq_len, d_model]
        # Add positional encodings (slice)
        
        pos = self.pos_embedding[:seq_len, :]
        x = x + pos  # [seq_len, d_model]

        x = x.unsqueeze(1)  # [seq_len, 1, d_model]

        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()  # True where to mask
        x = self.transformer(x, mask=mask)  # [seq_len, 1, d_model]
        x = x.squeeze(1)  # [seq_len, d_model]
        logits = self.output(x)  # [seq_len, vocab_size]
        log_probs = self.log_softmax(logits)  # [seq_len, vocab_size]
        return log_probs


# ---------------------------------------------------------------------
# NeuralLanguageModel wrapper implementing the LanguageModel API used by lm.py
# ---------------------------------------------------------------------
class NeuralLanguageModel(LanguageModel):
    def __init__(self, pytorch_model: NeuralLMModule, vocab_index: Indexer):
        """
        Wrap a trained pytorch module. We store the vocab_index for mapping chars <-> indices.
        """
        self.model = pytorch_model
        self.vocab_index = vocab_index
        # put model on CPU by default (lm.py probably runs on CPU). If GPU available, you can move to cuda.
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

            max_len = int(self.model.pos_embedding.shape[0])
            if len(idxs) > max_len:
                idxs = idxs[-max_len:]

            indices = torch.LongTensor(idxs).to(self.device)  
            log_probs = self.model(indices)                 
            return log_probs[-1].cpu().numpy()  
            

    def get_log_prob_sequence(self, next_chars:str, context:str) -> float:
        self.model.eval()
        total = 0.0
        ctx = context
        for ch in next_chars:
            lp = self.get_next_char_log_probs(ctx)
            idx = self.vocab_index.index_of(ch)
            if idx == -1:
                idx = self.vocab_index.index_of(' ')
            total += float(lp[idx])
            ctx += ch

        return total


# ---------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------
def train_lm(args, train_text, dev_text, vocab_index):
    """
    Train a NeuralLanguageModel on train_text and return the trained model wrapper.
    - train_text, dev_text: plain strings
    - vocab_index: Indexer mapping characters to indices (0..26)
    """
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    # Hyperparameters (chosen to balance speed and performance)
    device = torch.device("cpu")
    vocab_size = len(vocab_index)  # should be 27
    d_model = 128
    nhead = 8
    num_layers = 3
    dim_feedforward = 512
    dropout = 0.1
    max_len = 256
    seq_len = 64        # chunk length processed by model
    batch_size = 64     # how many chunks processed per optimizer step (we'll iterate sequentially to avoid memory issues)
    num_epochs = 4
    lr = 1e-3

    # Create PyTorch model
    pytorch_model = NeuralLMModule(vocab_size=vocab_size, d_model=d_model, nhead=nhead,
                                   num_layers=num_layers, dim_feedforward=dim_feedforward,
                                   dropout=dropout, max_len=max_len)
    pytorch_model.to(device)

    optimizer = optim.Adam(pytorch_model.parameters(), lr=lr)
    loss_fn = nn.NLLLoss()

    # Prepare indices for the full training text
    train_indices = [vocab_index.index_of(c) for c in train_text]
    # Replace unknowns (shouldn't be any) with space index
    unk_idx = vocab_index.index_of(' ')
    train_indices = [i if i >= 0 else unk_idx for i in train_indices]
    N = len(train_indices)

    # Create training chunks: overlapping windows with stride = seq_len (non-overlapping) to speed up.
    # We will produce chunks of length seq_len+1 (inputs and targets)
    def chunk_generator(indices, seq_len):
        i = 0
        while i + seq_len < len(indices):
            chunk = indices[i:i + seq_len + 1]
            if len(chunk) == seq_len + 1:
                yield chunk[:-1], chunk[1:]
            i += seq_len  # non-overlapping; you can set smaller stride to increase samples

    # Training loop
    pytorch_model.train()
    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        count = 0
        # iterate through chunks
        for X_chunk, Y_chunk in chunk_generator(train_indices, seq_len):
            # X_chunk, Y_chunk are lists of length seq_len
            X = torch.LongTensor(X_chunk).to(device)
            Y = torch.LongTensor(Y_chunk).to(device)
            optimizer.zero_grad()
            log_probs = pytorch_model(X)  # [seq_len, vocab_size]
            # NLLLoss expects [N, C], [N]
            loss = loss_fn(log_probs.view(-1, vocab_size), Y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pytorch_model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            count += 1
        avg_loss = total_loss / max(1, count)
        # quick dev perplexity monitoring (sample small windows from dev)
        # compute dev logprob on entire dev_text (might be short, 500 chars)
        wrapper_tmp = NeuralLanguageModel(pytorch_model, vocab_index)
        dev_logprob = wrapper_tmp.get_log_prob_sequence(dev_text, "")
        dev_ppl = math.exp(-dev_logprob / max(1, len(dev_text)))
        print(f"Epoch {epoch}/{num_epochs} - train_loss={avg_loss:.4f} - dev_perplexity={dev_ppl:.4f}")
    pytorch_model.eval()

    # Wrap into the LanguageModel API class and return
    nm = NeuralLanguageModel(pytorch_model, vocab_index)
    return nm