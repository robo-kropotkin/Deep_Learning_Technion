import torch
import pandas as pd
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig
from sklearn.model_selection import train_test_split
from tqdm import trange
from functions import preprocessW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig
from sklearn.model_selection import train_test_split
from torch import cuda

device = 'cuda' if cuda.is_available() else 'cpu'

# define parameters
MAX_LEN = 400
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 1
LEARNING_RATE = 1e-05
print("Loading tokenizer...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
criterion = torch.nn.CrossEntropyLoss()  # defining criterion

# unpack data
test_data = pd.read_parquet('data/test-00000-of-00001-35e9a9274361daed.parquet', engine='pyarrow')
train_data = pd.read_parquet('data/train-00000-of-00001-b943ea66e0040b18.parquet', engine='pyarrow')

labels = train_data["genre"].unique()

# pre process
train_data["genre"] = train_data["genre"].replace(labels, np.arange(len(labels)))
test_data["genre"] = test_data["genre"].replace(labels, np.arange(len(labels)))
sample = train_data["synopsis"][0]
train_data["synopsis"] = train_data["synopsis"].apply(preprocessW, args=(
400, tokenizer))  # convert to lower case and add symbols to data
test_data["synopsis"] = test_data["synopsis"].apply(preprocessW, args=(400, tokenizer))


# TODO: add data loader
# TODO: create CustomDataset
# TODO: add data loader


# Define data loader
class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.comment_text = dataframe["synopsis"]
        self.targets = dataframe["genre"]
        self.max_len = max_len

    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, index):
        comment_text = str(self.comment_text[index])
        comment_text = " ".join(comment_text.split())

        inputs = self.tokenizer.encode_plus(
            comment_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }


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
        _, output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output


model = MovieClassifier()
model.to(device)

optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)


# training function
def train(epoch):
    model.train()
    for batch_num, data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.float)

        outputs = model(ids, mask, token_type_ids)

        optimizer.zero_grad()
        loss = criterion(outputs, targets)
        if batch_num % 5000 == 0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# training
for epoch in range(EPOCHS):
    train(epoch)
