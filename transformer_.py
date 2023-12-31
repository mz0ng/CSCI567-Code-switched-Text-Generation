# -*- coding: utf-8 -*-
"""transformer.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1rRLlaRtEDPY7II4CvzmC3ZzMVHg7hlo3

*code for pretraining transformer referenced from https://github.com/chrishokamp/constrained_decoding*
"""

import copy
import json
import codecs
import logging

import numpy

from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math

import pandas as pd
import torch
from collections import Counter
from torchtext.data.utils import get_tokenizer
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import build_vocab_from_iterator
from typing import Iterable, List
import jieba

from collections import OrderedDict

en_tokens = set()
zh_tokens = set()

with open('/content/EN_tokens.txt', 'r') as file:
    for line in file:
        en_tokens.add(line.strip())

with open('/content/ZH_tokens.txt', 'r') as file:
    for line in file:
        zh_tokens.add(line.strip())

# Creating a disjoint union
tokens = en_tokens.union(zh_tokens)

# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3

# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

# Start building the vocabulary with special symbols
vocab = OrderedDict((symbol, idx) for idx, symbol in enumerate(special_symbols))

# Mapping each word to a unique index
vocab.update((token, idx+len(special_symbols)) for idx, token in enumerate(tokens))

# Representation for whitespace
whitespace_token = ' '

# Check if whitespace is not already in the vocabulary
if whitespace_token not in vocab:
    # Assign the next available index to the whitespace character
    vocab[whitespace_token] = len(vocab)

from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

torch.manual_seed(0)

VOCAB_SIZE = len(vocab)
EMB_SIZE = 768
NHEAD = 12
FFN_HID_DIM = 512
BATCH_SIZE = 128
NUM_ENCODER_LAYERS = 8
NUM_DECODER_LAYERS = 8

transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, VOCAB_SIZE, FFN_HID_DIM)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import pandas as pd
from typing import Iterable, List
from torch.utils.data import Dataset, DataLoader

class Parallel_Dataset(Dataset):
    def __init__(self, csv_file):
        self.dataframe = pd.read_csv('/content/MT_train.csv')

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input = self.dataframe.iloc[idx, 0]
        output = self.dataframe.iloc[idx, 1]

        sample = {'input': input, 'output': output}

        return sample

!pip install sacremoses
!pip3 install jieba
from torch.nn.utils.rnn import pad_sequence
from sacremoses import MosesTokenizer
import jieba

# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(tokens: List[str]):
    tokenids = []
    # tokenids = [vocab.get(token, vocab[UNK_IDX]) for token in tokens]
    for token in tokens:
        if token not in vocab:
            tokenids.append(vocab['<unk>'])
        else:
            tokenids.append(vocab[token])
    input_tensor = torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(tokenids),
                      torch.tensor([EOS_IDX])))
    return input_tensor

token_transform = {}
en_tokenizer = MosesTokenizer(lang='en')

def tokenize_en(sentence):
    return en_tokenizer.tokenize(sentence, escape=False)

def tokenize_zh(sentence):
    # This collects the actual tokens and ignores other information that jieba's tokenizer yields
    return [token for token, start, stop in jieba.tokenize(sentence)]

token_transform[0] = tokenize_en
token_transform[1] = tokenize_zh


# ``src`` and ``tgt`` language text transforms to convert raw strings into tensors indices
text_transform = {}
for ln in [0, 1]:
    text_transform[ln] = sequential_transforms(token_transform[ln], #Tokenization
                                                #Numericalization
                                               tensor_transform) # Add BOS/EOS and create tensor


# function to collate data samples into batch tensors
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    # for sample in batch:
    #   src_sample = sample[0]
    #   tgt_sample = sample[1]
    for p in range(0, 128, 4):
        src_batch.append(text_transform[0](batch[p]['input'].rstrip("\n")))
        src_batch.append(text_transform[0](batch[p+1]['input'].rstrip("\n")))
        src_batch.append(text_transform[1](batch[p+2]['input'].rstrip("\n")))
        src_batch.append(text_transform[1](batch[p+3]['input'].rstrip("\n")))

        tgt_batch.append(text_transform[0](batch[p+0]['output'].rstrip("\n")))
        tgt_batch.append(text_transform[1](batch[p+1]['output'].rstrip("\n")))
        tgt_batch.append(text_transform[0](batch[p+2]['output'].rstrip("\n")))
        tgt_batch.append(text_transform[1](batch[p+3]['output'].rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch

from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.optim as optim


def train_epoch(model, optimizer):
    model.train()
    losses = 0
    train_iter = Parallel_Dataset(csv_file='/content/MT_train.csv')
    train_dataloader = DataLoader(train_iter, batch_size=128)

    for batch in train_dataloader:


    for src, tgt in train_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(list(train_dataloader))

from timeit import default_timer as timer
NUM_EPOCHS = 18


for epoch in range(1, NUM_EPOCHS+1):
    start_time = timer()
    train_loss = train_epoch(transformer, optimizer)
    end_time = timer()

!pip install langid

import langid

idx_to_token = {idx: token for token, idx in vocab.items()}


def token_to_string(token_idx):
    """ Convert a token index to its string representation. """
    return idx_to_token.get(token_idx, '<unk>')

class BeamHypothesis:
    def __init__(self, tokens, score, num_switches):
        self.tokens = tokens  # Token sequence
        self.score = score    # Cumulative score
        self.num_switches = num_switches  # Number of code-switches

    def extend(self, token, score, switch):
        return BeamHypothesis(
            tokens=self.tokens + [token],
            score=self.score + score,
            num_switches=self.num_switches + (1 if switch else 0)
        )

    def latest_token(self):
        return self.tokens[-1]

    def is_codeswitch(self, new_token):
        """
        Determine if adding `new_token` to the current hypothesis represents a code-switch.

        Args:
            new_token (int): Index of the new token.

        Returns:
            bool: True if adding `new_token` is a code-switch, False otherwise.
        """
        if len(self.tokens) == 0:
            return False

        current_token_str = token_to_string(self.tokens[-1])
        new_token_str = token_to_string(new_token)

        current_token_lang = langid.classify(current_token_str)[0]
        new_token_lang = langid.classify(new_token_str)[0]

        return current_token_lang != new_token_lang


def grid_beam_search(model, src, src_mask, max_length, beam_size, switch_threshold):
    # Move src and src_mask to the appropriate device
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    # Initialize beam with the start-of-sequence token
    start_token = BOS_IDX
    initial_hypothesis = BeamHypothesis([start_token], 0.0, 0)
    beam = [initial_hypothesis]
    memory = model.encode(src, src_mask).to(DEVICE)
    for step in range(max_length):
        next_beam = []
        for hypothesis in beam:
            if hypothesis.latest_token() == EOS_IDX:
                # If EOS token is reached, add hypothesis to the next beam directly
                next_beam.append(hypothesis)
                continue

            # Get the probability distribution for the next token
            tgt_tokens = torch.LongTensor(hypothesis.tokens).unsqueeze(1).to(DEVICE)
            tgt_mask = generate_square_subsequent_mask(tgt_tokens.size(0)).to(DEVICE)
            out = model.decode(tgt_tokens, memory, tgt_mask)  # Assuming `memory` is already computed
            probs = model.generator(out[-1])


            # Consider top-k tokens for the next step
            topk_probs, topk_tokens = probs.topk(beam_size)
            for i in range(beam_size):
                token = topk_tokens[0][i].item()  # Index into the second dimension to get a scalar tensor
                score = topk_probs[0][i].item()   # Same as above
                switch = hypothesis.is_codeswitch(token)
                next_beam.append(hypothesis.extend(token, score, switch)) # item() converts to a Python float

        # Sort hypotheses in the beam based on score and keep top `beam_size` hypotheses
        # Also ensure that there are hypotheses with enough code-switches
        beam = sorted(next_beam, key=lambda h: (h.score, h.num_switches >= switch_threshold))
        beam = beam[:beam_size]
    return beam

src_text = "come up with yourself"
src_tokens = src_text.split()[:20]  # Tokenize and truncate/pad to 20 tokens
src_indices = [vocab.get(token, UNK_IDX) for token in src_tokens]
src = torch.tensor(src_indices).unsqueeze(1)

src_mask = (src != PAD_IDX).transpose(0, 1)

# Assuming PAD_IDX is defined somewhere above.
# We want src_mask to be False where src is PAD_IDX and True elsewhere
# Also, the mask needs to be square, so we repeat it `src.size(0)` times.
src_mask = (src != PAD_IDX)
src_mask = src_mask.repeat(src.size(0), 1)
src_mask = src_mask.reshape(src.size(0), src.size(0)).transpose(0, 1).to(DEVICE)

# rest of your code

# Now the shape of src_mask should be correct
print("src_mask shape:", src_mask.shape)

print(DEVICE)

transformer.eval()

max_length = 58
beam_size = 10
switch_threshold = 7

src = src.to(DEVICE)
src_mask = src_mask.to(DEVICE)

print("src shape:", src.shape)
print("src_mask shape:", src_mask.shape)

# Check the values of src_mask for debugging
print("src_mask values:", src_mask)

best_hypotheses = grid_beam_search(transformer, src, src_mask, max_length, beam_size, switch_threshold)

best_hypothesis = max(best_hypotheses, key=lambda h: h.score if h.num_switches >= switch_threshold else float('-inf'))
best_sequence = best_hypothesis.tokens

best_words = [token_to_string(token) for token in best_sequence if token not in (EOS_IDX, BOS_IDX)]

best_words