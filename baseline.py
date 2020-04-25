#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch.utils.data as utils
import torch
import torch.nn as nn
import torch.optim as optim
from torch.hub import load
import torchvision.transforms as transforms

from sklearn.metrics import f1_score, accuracy_score

import pandas as pd
import matplotlib.pyplot as plt
import math
from matplotlib.backends.backend_pdf import PdfPages

import os

# In[2]:

class TweetDataset(utils.Dataset):
    def __init__(self, path='Tweets.csv'):
        data = pd.read_csv(path)
        self.txt = data.Tweets.to_list()
        self.lbl = data.Labels.to_list()
        
    def __getitem__(self, index):
        #supporting fetching a data sample for a given key
        return self.txt[index], self.lbl[index]
    
    def __len__(self):
        #return the size of the dataset
        return len(self.txt)


# In[3]:


class net(nn.Module):
	def __init__(self, tokenizer, model, extract_method):
		super(net, self).__init__()
		self.model = model
		self.extract_method = extract_method
		self.tokenizer = tokenizer if tokenizer else lambda x: x
		self.m = nn.AvgPool1d(3, stride=2)
		# in_feature sizes
		# Bert: (batch_size, sequence_length, hidden_size)
		# 
		self.linear = nn.Linear(in_features=5*383, out_features=3, bias=True)
	def forward(self, x):
		y=[]
		for data in test_data:
			input_ids = torch.tensor(self.tokenizer.encode(data)).unsqueeze(0)
			last_hidden_states = self.model(input_ids)[0]
			pooled = self.m(last_hidden_states)
			y.append(self.linear(x_hat))
		return torch.tensor(y)


# In[4]:

path='Tweets.csv'

dataset = TweetDataset(path)
train_ration = 0.7
train_dataset, test_dataset = utils.random_split(dataset, [int(len(dataset)*0.7), len(dataset)-int(len(dataset)*0.7)])

train_loader = utils.DataLoader(train_dataset)
test_loader = utils.DataLoader(test_dataset)

cache = pd.read_csv(path)
vocab = set()
cache.Tweets.str.lower().str.split().apply(vocab.update)
cache = None

# In[7]:

# model creation
'''
os.system('git clone https://github.com/huggingface/transformers')
os.system('git clone https://github.com/pytorch/fairseq')
os.system('git clone https://github.com/zihangdai/xlnet/')
'''

from transformers import BertModel, BertConfig, BertTokenizer, RobertaTokenizer
# from fairseq.fairseq.models.roberta import RobertaModel
# from transformers import XLNetTokenizer, XLNetModel

bert_config = BertConfig()
bert_model = BertModel(bert_config)
bert_model.eval()
bert_result = lambda x: x[0]
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

roberta_model = load('pytorch/fairseq', 'roberta.large.mnli')
roberta_model.eval()
roberta_result = roberta_model.extract_features
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

models = {
    'Bert': net(bert_tokenizer, bert_model, bert_result),
    'Roberta': net(roberta_tokenizer, roberta_model, roberta_result),
    # 'XLNet': net(xlnet_model, xlnet_result)
}


# In[ ]:


def evalute(model, test_loader, device):
    y_true, y_pred = [], []

    model.eval()
    for data_batch, batch_labels in train_loader:
        preds = model(data_batch).cpu()
        y_pred.extend(list(preds))
        y_true.extend(list(batch_labels))

    accuracy = accuracy_score(y_true, y_pred, normalize=True)
    f1 = f1_score(y_true, y_pred)
    
    return accuracy, f1


# In[ ]:


# training and evaluation
nrows, ncols, index = math.ceil(len(models.keys())), 2, 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EPOCHS = 50
x = list(range(1, NUM_EPOCHS+1))
pdf = ('Baseline.pdf')
fig, axs = plt.subplot(nrows, ncols)

for key in models.keys():
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    model = models[key]
    model.to(device)
    
    accuracies, f1_scores = [], []
    
    for epoch in range(NUM_EPOCHS):
        # training
        model.train()
        for i, (data_batch, batch_labels) in enumerate(train_loader):
            preds = model(data_batch)
            loss = criterion(preds, batch_labels.to(device))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # evaluation
        model.eval()
        accuracy, f1 = evaluate(model, test_loader, device)
        accuracies.append(accuracy)
        f1_scores.append(f1)
    
    torch.save(model.state_dict(), '{}_baseline.pt'.format(key))

    axs[index, 0].set_title('Accuracy vs Epochs for {}'.format(key.lower().capitalize()))
    axs[index, 0].set_xlabel('Number of Epochs Trained')
    axs[index, 0].set_ylabel('Test Accuracy on Sentiment Classification Task')
    axs[index, 0].plot(x, accuracies)
    
    axs[index, 1].set_title('F1-Score vs Epochs for {}'.format(key.lower().capitalize()))
    axs[index, 1].set_xlabel('Number of Epochs Trained')
    axs[index, 1].set_ylabel('Test F1-Score on Sentiment Classification Task')
    axs[index, 1].plot(x, f1_scores)
    
    index += 1

plt.show()
pdf.savefig(fig)

# In[ ]:




