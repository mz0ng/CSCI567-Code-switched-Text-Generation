#pip install langid

import langid

import pandas as pd
import os
import torch
from collections import Counter
from torchtext.data.utils import get_tokenizer
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import build_vocab_from_iterator
from typing import Iterable, List
import jieba

from collections import OrderedDict
from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math
DEVICE = torch.device('cuda')



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

src_text = "Testing for a string"
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


with open('weights_token.pt', 'rb') as f:
     transformer = torch.load(f)

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