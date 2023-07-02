import torch
import pandas as pd
import numpy as np
import random
from tqdm import trange
from functions import preprocessW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel , BertConfig
from sklearn.model_selection import train_test_split
from torch import cuda


device = 'cuda' if cuda.is_available() else 'cpu'
print(device)


# unpack data
test_data = pd.read_parquet('data/test-00000-of-00001-35e9a9274361daed.parquet', engine='pyarrow')
train_data = pd.read_parquet('data/train-00000-of-00001-b943ea66e0040b18.parquet', engine='pyarrow')
x_train = train_data["synopsis"]
y_train = train_data["genre"]
x_test = test_data["synopsis"]
y_test = test_data["genre"]
labels = y_train.unique()
y_train = y_train.replace(labels, np.arange(len(labels)))
criterion = torch.nn.CrossEntropyLoss() # defining criterion

# pre process
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') # define pre-trained tokenizer
sample = x_train[0]
x_train = x_train.apply(preprocessW, args=(400, tokenizer)) # convert to lower case and add symbols to data
x_test = x_test.apply(preprocessW, args=(400, tokenizer))


# TODO: add data loader
# TODO: create CustomDataset
# TODO: add data loader


# Creating the class moodel to fine tune and passing it to device
class MovieClassifier(torch.nn.Module): 
    def __init__(self):
        super(MovieClassifier, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 10)
    
    def forward(self, ids, mask, token_type_ids):
        _, output_1= self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output


model = MovieClassifier()
model.to(device)
