"""
COMS 4705 Natural Language Processing Spring 2021
Kathy McKeown
Homework 2: Emotion Classification with Neural Networks - Models File

<YOUR NAME HERE>
<YOUR UNI HERE>
"""

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F

class DenseNetwork(nn.Module):
    def __init__(self, embeddings):
        super(DenseNetwork, self).__init__()

        ########## YOUR CODE HERE ##########
        # create embedding layer and load pretrained
        self.embedding_layer = nn.Embedding(embeddings.shape[0], embeddings.shape[1], padding_idx=0)
        # self.embedding_layer.weight.data = (torch.Tensor(embeddings)).float()
        self.embedding_layer.weight = nn.Parameter(embeddings)
        # do not fine tune embeddings
        self.embedding_layer.weight.requires_grad = False
        self.linear1=nn.Linear(100, 64)
        self.linear2=nn.Linear(64, 64)
        self.clf=nn.Linear(64, 4)


    def forward(self, x):
        ########## YOUR CODE HERE ##########
        logits=self.embedding_layer(x)
        # make sure no error
        logits=logits.float()
        # avg pooling
        num_embedded = (x == self.embedding_layer.padding_idx) \
            .float().sum(dim=1).clamp(1)
        logits = logits.sum(dim=1) / num_embedded.view(-1,1)
        # max pooling
        # logits,_ = logits.max(dim=1)
        # pass into first layer
        logits=self.linear1(logits)
        logits = F.relu(F.dropout(logits, p=0.1))
        # pass into second layer
        logits=self.linear2(logits)
        logits = F.relu(F.dropout(logits, p=0.1))
        # softmax
        logits=self.clf(logits)
        return F.softmax(logits, dim=-1)
        


class RecurrentNetwork(nn.Module):
    def __init__(self,embeddings):
        super(RecurrentNetwork, self).__init__()

        ########## YOUR CODE HERE ##########
        # create embedding layer and load pretrained
        self.embedding_layer = nn.Embedding(embeddings.shape[0], embeddings.shape[1], padding_idx=0)
        self.embedding_layer.weight = nn.Parameter(embeddings)
        self.embedding_layer.weight.requires_grad = False
        # initialize rnn
        self.rnn=nn.RNN(100, 64, 2, nonlinearity="relu",bidirectional=True)
        self.linear1=nn.Linear(128, 64)
        self.clf=nn.Linear(64, 4)
        
        # self.clf=nn.Linear(128, 4)
    # x is a PaddedSequence for an RNN
    def forward(self, x):
        ########## YOUR CODE HERE ##########
        logits=self.embedding_layer(x)
        logits=logits.float()
        logits,_=self.rnn(logits)
        # maxpool
        logits,_ = logits.max(dim=1)
        logits=self.linear1(logits)
        logits=self.clf(logits)
        return F.softmax(logits, dim=-1)

################# extension-grading ######################
"""
Build a single layer bi-LSTM with a single fully connected layer on top of it
"""
class ExperimentalNetwork(nn.Module):
    def __init__(self, embeddings):
        super(ExperimentalNetwork, self).__init__()
        # load pretrained embeddings
        self.embedding_layer = nn.Embedding(embeddings.shape[0], embeddings.shape[1], padding_idx=0)
        self.embedding_layer.weight = nn.Parameter(embeddings)
        self.embedding_layer.weight.requires_grad = False
        # initialize BiLSTM
        self.bilstm = nn.LSTM(embeddings.shape[1],64,1, bidirectional=True)
        self.clf=nn.Linear(128, 4)
        

        ########## YOUR CODE HERE ##########

    # x is a PaddedSequence for an RNN
    def forward(self, x):
        ########## YOUR CODE HERE ##########
        logits=self.embedding_layer(x)
        logits=logits.float()
        logits,_=self.bilstm(logits)
        # get sum and normalize
        # num_embedded = (x == self.embedding_layer.padding_idx) \
        #     .float().sum(dim=1).clamp(1)
        # logits = logits.sum(dim=1) / num_embedded.view(-1,1)
        # maxpool
        logits,_ = logits.max(dim=1)
        logits=self.clf(logits)
        return logits