import pandas as pd
import torch
import numpy as np

test_data = pd.read_parquet('data/test-00000-of-00001-35e9a9274361daed.parquet', engine='pyarrow')
train_data = pd.read_parquet('data/train-00000-of-00001-b943ea66e0040b18.parquet', engine='pyarrow')
x_train = train_data["synopsis"]
y_train = train_data["genre"]
x_test = test_data["synopsis"]
y_test = test_data["genre"]
labels = y_train.unique()
y_train = y_train.replace(labels, np.arange(len(labels)))
criterion = torch.nn.CrossEntropyLoss()
random_labels = torch.ones((len(x_train), len(labels)))
loss = criterion(random_labels, torch.tensor(y_train))
print(loss)