"""
COMS 4705 Natural Language Processing Spring 2021
Kathy McKeown
Homework 2: Emotion Classification with Neural Networks - Main File

<Weiyao Xie>
<wx2251>
"""

# Imports
import nltk
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse

# Imports - our files
import utils
import models
torch.manual_seed(0)
# Global definitions - data
DATA_FN = 'data/crowdflower_emotion.csv'
LABEL_NAMES = ["happiness", "worry", "neutral", "sadness"]

# Global definitions - architecture
EMBEDDING_DIM = 100  # We will use pretrained 100-dimensional GloVe
BATCH_SIZE = 128
NUM_CLASSES = 4
USE_CUDA = torch.cuda.is_available()  # CUDA will be available if you are using the GPU image for this homework

# Global definitions - saving and loading data
FRESH_START = False  # set this to false after running once with True to just load your preprocessed data from file
#                     (good for debugging)
TEMP_FILE = "temporary_data.pkl"  # if you set FRESH_START to false, the program will look here for your data, etc.


def train_model(model, loss_fn, optimizer, train_generator, dev_generator):
    """
    Perform the actual training of the model based on the train and dev sets.
    :param model: one of your models, to be trained to perform 4-way emotion classification
    :param loss_fn: a function that can calculate loss between the predicted and gold labels
    :param optimizer: a created optimizer you will use to update your model weights
    :param train_generator: a DataLoader that provides batches of the training set
    :param dev_generator: a DataLoader that provides batches of the development set
    :return model, the trained model
    """
    ########## YOUR CODE HERE ##########
    # TODO: Given a model, data, and loss function, you should do the following:
    # TODO: 1) Loop through the whole train dataset performing batch optimization with the optimizer of your choice,
    running_loss=0
    dev_loss=2**31
    max_acc=0
    earlystop_patience=0
    for epoch_num in range(15):
        model.train()
        for i, data in enumerate(train_generator, 0):
            # get training data and labels from trainloader
            # clean the old gradients
            # forward prop
            # get the loss of this batch
            # backprop
            inputs, labels = data
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            # update parameters
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss.item()

    
        # do not need to update parameter
        with torch.no_grad():
            dev_loss_current=0
            all_preds = []
            labels = []
            # get the sum of loss for all batches
            for i, (inputs_dev, labels_dev) in enumerate(dev_generator, 0):
                # get prediction
                outputs_dev = model(inputs_dev)
                # get loss and update the total loss
                loss = loss_fn(outputs_dev, labels_dev)
                dev_loss_current+=loss
                # # for accuracy
                # # just a test I want to see
                # labels+=list(labels_dev)
                # all_preds.append(outputs_dev.numpy())
                
            
            print("The dev loss of {}th epoch is: {}".format(epoch_num,dev_loss_current))
            # final_preds =  np.concatenate(all_preds, axis=0)
            # curr_accuracy = (labels==(final_preds.argmax(-1))).mean()
            # print("validation accuracy is:", curr_accuracy)

        # update patience if loss is larger than before
        if dev_loss_current >= dev_loss:
            
            earlystop_patience+=1
        # zero the patience if loss is lower than previous epoch
        else:
            earlystop_patience=0
            dev_loss=dev_loss_current
        # max_acc=curr_accuracy
        # update loss
        # stop training since there is no improvement
        if earlystop_patience>=3:
            print("There is no improvement on the model, so early stop is triggered.")
            print("Training finished")
            print("The dev loss is {}".format(dev_loss_current))
            return model
    print("Training finished")
    print("The dev loss is {}".format(dev_loss_current))
    return model



def test_model(model, loss_fn, test_generator):
    """
    Evaluate the performance of a model on the development set, providing the loss and macro F1 score.
    :param model: a model that performs 4-way emotion classification
    :param loss_fn: a function that can calculate loss between the predicted and gold labels
    :param test_generator: a DataLoader that provides batches of the testing set
    """
    gold = []
    predicted = []

    # Keep track of the loss
    loss = torch.zeros(1)  # requires_grad = False by default; float32 by default
    if USE_CUDA:
        loss = loss.cuda()

    model.eval()

    # Iterate over batches in the test dataset
    with torch.no_grad():
        for X_b, y_b in test_generator:
            # Predict
            y_pred = model(X_b)

            # Save gold and predicted labels for F1 score - take the argmax to convert to class labels
            gold.extend(y_b.cpu().detach().numpy())
            predicted.extend(y_pred.argmax(1).cpu().detach().numpy())

            loss += loss_fn(y_pred.double(), y_b.long()).data

    # Print total loss and macro F1 score
    print("Test loss: ")
    print(loss)
    print("F-score: ")
    print(f1_score(gold, predicted, average='macro'))


def main(args):
    """
    Train and test neural network models for emotion classification.
    """
    # Prepare the data and the pretrained embedding matrix
    if FRESH_START:
        print("Preprocessing all data from scratch....")
        train, dev, test = utils.get_data(DATA_FN)
        # train_data includes .word2idx and .label_enc as fields if you would like to use them at any time
        train_generator, dev_generator, test_generator, embeddings, train_data = utils.vectorize_data(train, dev, test,
                                                                                                BATCH_SIZE,
                                                                                                EMBEDDING_DIM)

        print("Saving DataLoaders and embeddings so you don't need to create them again; you can set FRESH_START to "
              "False to load them from file....")
        with open(TEMP_FILE, "wb+") as f:
            pickle.dump((train_generator, dev_generator, test_generator, embeddings, train_data), f)
    else:
        try:
            with open(TEMP_FILE, "rb") as f:
                print("Loading DataLoaders and embeddings from file....")
                train_generator, dev_generator, test_generator, embeddings, train_data = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("You need to have saved your data with FRESH_START=True once in order to load it!")
    

    if args.model == "dense":

    # Use this loss function in your train_model() and test_model()
        loss_fn = nn.CrossEntropyLoss()
        ########## YOUR CODE HERE ##########    
        # FCN
        model_dense=models.DenseNetwork(embeddings)
        # intialize optimizer
        optimizer = optim.Adam(model_dense.parameters(), lr=0.01)
        print("Training DenseNetwork")
        # intialize densenetwork model
        model_dense = train_model(model_dense, loss_fn, optimizer, train_generator, dev_generator)
        print("Testing DenseNetwork")
        # get test report
        test_model(model_dense, loss_fn, test_generator)
    elif args.model == "RNN":

        loss_fn = nn.CrossEntropyLoss()
        # # vanilla RNN
        model_rnn=models.RecurrentNetwork(embeddings)
        optimizer = optim.Adam(model_rnn.parameters(), lr=0.003)
        print("Training RecurrentNetwork")
        model_rnn = train_model(model_rnn, loss_fn, optimizer, train_generator, dev_generator)
        print("Testing RecurrentNetwork")
        test_model(model_rnn, loss_fn, test_generator)

################# extension-grading ######################
    elif args.model == "extension1":
        # bidiretional LSTM
        loss_fn = nn.CrossEntropyLoss()
        model_lstm=models.ExperimentalNetwork(embeddings)
        optimizer = optim.Adam(model_lstm.parameters(), lr=0.003)
        print("Training Extension1")
        model_lstm = train_model(model_lstm, loss_fn, optimizer, train_generator, dev_generator)
        print("Testing Extension1")
        test_model(model_lstm, loss_fn, test_generator)

################# extension-grading ######################
    elif args.model == "extension2":
        # get the data with new preprocessed content
        # please see utils for implementation
        print("Extension2 will rerun the data preprocessing part")
        train, dev, test = utils.get_data_ext(DATA_FN)
        train_generator, dev_generator, test_generator, embeddings, train_data = utils.vectorize_data(train, dev, test,
                                                                                                BATCH_SIZE,
                                                                                                EMBEDDING_DIM)
        # use same model as extension1                                                                                        
        loss_fn = nn.CrossEntropyLoss()
        model_lstm=models.ExperimentalNetwork(embeddings)
        optimizer = optim.Adam(model_lstm.parameters(), lr=0.003)
        print("Training Extension2")
        model_lstm = train_model(model_lstm, loss_fn, optimizer, train_generator, dev_generator)
        print("Testing Extension2")
        test_model(model_lstm, loss_fn, test_generator)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model', required=True,
                        choices=["dense", "RNN", "extension1", "extension2"],
                        help='The name of the model to train and evaluate.')
    args = parser.parse_args()
    main(args)
