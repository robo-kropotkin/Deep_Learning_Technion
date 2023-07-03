import torch
import pandas as pd
import numpy as np
import sys
import random
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig
from sklearn.model_selection import train_test_split
from functions import preprocessW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig
from sklearn.model_selection import train_test_split
from torch import cuda

device = 'cuda' if cuda.is_available() else 'cpu'
tqdm.pandas()

# define parameters
MAX_LEN = 400
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 4
LEARNING_RATE = 1e-5
print("Loading tokenizer...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
criterion = torch.nn.CrossEntropyLoss()  # defining criterion

# unpack data
test_data = pd.read_parquet('data/test-00000-of-00001-35e9a9274361daed.parquet', engine='pyarrow')
train_data = pd.read_parquet('data/train-00000-of-00001-b943ea66e0040b18.parquet', engine='pyarrow')
if 'debug' in sys.argv:
    test_data = test_data[:10]
    train_data = train_data[:100]

labels = train_data["genre"].unique()

# pre process
train_data["genre"] = train_data["genre"].replace(labels, np.arange(len(labels)))
test_data["genre"] = test_data["genre"].replace(labels, np.arange(len(labels)))
sample = train_data["synopsis"][0]
train_data["synopsis"] = train_data["synopsis"].apply(preprocessW, args=(
400, tokenizer))  # convert to lower case and add symbols to data
test_data["synopsis"] = test_data["synopsis"].apply(preprocessW, args=(400, tokenizer))


# Define data loader
class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.synopsi = dataframe["synopsis"]
        self.targets = dataframe["genre"]
        self.max_len = max_len

    def __len__(self):
        return len(self.synopsi)

    def __getitem__(self, index):
        inputs = self.tokenizer.encode_plus(self.synopsi[index])
        ids = inputs['input_ids']
        mask = np.array(ids) != 0
        token_type_ids = inputs["token_type_ids"]

        return torch.tensor(ids, dtype=torch.long).to(device), \
               torch.tensor(mask, dtype=torch.long).to(device), \
               torch.tensor(token_type_ids, dtype=torch.long).to(device), \
               torch.tensor(self.targets[index], dtype=torch.long).to(device)


training_set = CustomDataset(train_data, tokenizer, MAX_LEN)
testing_set = CustomDataset(test_data, tokenizer, MAX_LEN)

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
               'shuffle': True,
               'num_workers': 0
               }

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)


# Creating the class model to fine tune and passing it to device
class MovieClassifier(torch.nn.Module):
    def __init__(self):
        super(MovieClassifier, self).__init__()
        self.l1 = BertModel.from_pretrained('bert-base-uncased')
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 10)

    def forward(self, ids, mask, token_type_ids):
        output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids).pooler_output
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output


class warmup_scheduler(torch.optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer, warmup_steps, last_epoch=-1, verbose=False):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch=-1, verbose=False)

    def get_lr(self):
        return [group['lr'] * min(self._step_count / self.warmup_steps ** (-1.5), self._step_count ** (-0.5)) for group
                in self.optimizer.param_groups]


model = MovieClassifier()
model.to(device)

optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
scheduler = warmup_scheduler(optimizer, warmup_steps=4000)


# training function
def train(epoch):
    model.train()
    for batch_num, data in enumerate(training_loader, 0):
        ids, mask, token_type_ids, targets = data

        outputs = model(ids, mask, token_type_ids)

        optimizer.zero_grad()
        loss = criterion(outputs, targets)
        if batch_num % 500 == 0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# training
for epoch in range(EPOCHS):
    train(epoch)
