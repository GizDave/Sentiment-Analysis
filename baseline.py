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
    def __init__(self, transformer):
        super(net, self).__init__()
        self.transformer = transformer
        self.linear = nn.Linear(in_features=..., out_features=2, bias=True)
        
    def forward(self, x):
        x_hat = self.transformer(x)
        y = self.linear(x_hat)
        return y


# In[4]:


dataset = TweetDataset()
train_ration = 0.7
train_dataset, test_dataset = utils.random_split(dataset, [int(len(dataset)*0.7), len(dataset)-int(len(dataset)*0.7)])

train_loader = utils.DataLoader(train_dataset)
test_loader = utils.DataLoader(test_dataset)


# In[7]:


# model creation
models = {
    'Bert': net(load('pytorch/fairseq', 'model', 'bert-base', pretrained=True)),
    'Roberta': net(load('pytorch/fairseq', 'model', 'roberta.base', pretrained=True)),
    'XLNet': net(load('pytorch/fairseq', 'model', 'xlnet.base', pretrained=True))
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




