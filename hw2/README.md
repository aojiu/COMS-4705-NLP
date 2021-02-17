# Homework 2: Emotion Classification with Neural Networks
Natural Language Processing

Prof Kathleen McKeown

Weiyao Xie (wx2251)

## Getting Started
### Software Requirement
- Python 3.7
- Numpy >= 1.18.1.
- sklearn 0.22.2
- NLTK 3.5
- pandas 1.0.3



### Usage
Run hw2.py to train the selected model and get the test results.
There are four required arguments to pass in.
```
python hw2.py --model <model name> 
```

## hw2.py: 
* This is the main function.
* First, functions in utils.py will preprocess the raw data and make a local copy. After running the code for the first time, set ```FRESH_START = False``` to save time.
* Then, read training and test data.
* Initilize loss function ```loss_fn=nn.CrossEntropyLoss()``` and optimizer ```optimizer = optim.Adam(model.parameters(), lr=0.01)``` for each model.
* Use ```train_model``` to get the trained model, and report f1 score using ```test_model```.

## model.py: 
* The file contains all models used in hw1.py
* Implementation follows the instruction in homework
* There is only one extension in this file.
* See extension2 in utils.py
* RNN is a bidirectional RNN ```self.rnn=nn.RNN(100, 64, 2, nonlinearity="relu",bidirectional=True)```.
* Extension is a bi-LSTM with a step learning rate scheduler```self.bilstm = nn.LSTM(embeddings.shape[1],64,1, bidirectional=True)```

## utils.py: 
* The file contains code used in preprocessing and generating embeddings.
* Extension2 is included in this file.
* Added two extra preprocessing functions to convert urls into |||URL|| tags and also convert usernames into @||user||.




