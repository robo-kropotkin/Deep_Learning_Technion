## imports
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

from tqdm import trange
import random
from functions import preprocessW

## unpack data 
test_data = pd.read_parquet('data/test-00000-of-00001-35e9a9274361daed.parquet', engine='pyarrow')
train_data = pd.read_parquet('data/train-00000-of-00001-b943ea66e0040b18.parquet', engine='pyarrow')
x_train = train_data["synopsis"]
y_train = train_data["genre"]
x_test = test_data["synopsis"]
y_test = test_data["genre"]
labels = y_train.unique()
y_train = y_train.replace(labels, np.arange(len(labels)))
criterion = torch.nn.CrossEntropyLoss()# defining criterion
if(1):# testing criterion
    random_labels = torch.ones((len(x_train), len(labels)))
    loss = criterion(random_labels, torch.tensor(y_train))
    print(loss)

##pre process
x_train = x_train.apply(preprocessW)# convert to lower case and add symbols to data 
x_test = x_test.apply(preprocessW)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') # define pre-trained tokenizer

print(x_train[0])