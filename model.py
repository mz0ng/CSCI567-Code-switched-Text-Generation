from googletrans import Translator
from typing import Tuple
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
import pickle
import csv
import time
import math
import torch.nn.functional as F
import sys
import llm



### Transformer model
### credits to reference source: https://github.com/pytorch/examples/tree/main
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Transformer):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__(d_model=ninp, nhead=nhead, dim_feedforward=nhid, num_encoder_layers=nlayers)
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)

        self.input_emb = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ninp)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        return torch.log(torch.tril(torch.ones(sz,sz)))

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.input_emb.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.input_emb(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.encoder(src, mask=self.src_mask)
        output = self.decoder(output)
        return F.log_softmax(output, dim=-1)


### load word embeddings
with open('Embeddings', "rb") as f:
      data = pickle.load(f)

with torch.no_grad():
    torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

### create new model instance
model = TransformerModel(ntoken=1682, ninp=1536, nhead=12, nhid=200, nlayers=8, dropout=0.5).to(device)

### retrieve saved model for further use
# with open('model.pt', 'rb') as f:
#     model = torch.load(f)


total_loss = 0
log_interval = 20
epochs = 5
batch_size = 4
lr = 10
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

### training the model
model.train()
count=0
for epoch in range(epochs):
    print('Start of Epoch')
    ### due to limited resources, we can only train the model pair by pair
    for i in range(len(data)):
        if max(len(data[i][0]),len(data[i][1])) < 10 and len(data[i][0])==len(data[i][1]):
            count += 1
            xbatch = torch.tensor(data[i][0]).long().to(device)
            ybatch = torch.tensor(data[i][1]).long().to(device)
            output = model(xbatch)
            output_flat = output.view(-1, 1536)
            optimizer.zero_grad()
            loss = criterion(output, ybatch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            print('Batch loss: ', loss.item())
            total_loss += loss.item()
            print('Total loss: ', total_loss)
            if (i+1) % log_interval == 0 and i > 0:
                lr = scheduler.get_last_lr()[0]
                print(f'| epoch {epoch:3d} | {i:5d}/{2500:5d} batches | '
                    f'lr {lr:02.2f}')
                total_loss = 0
                start_time = time.time()

### save trained model
with open('model.pt', 'wb') as f:
        torch.save(model, f) 
