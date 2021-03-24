# Homework 2: Adventures with word embeddings
Natural Language Processing

Prof Kathleen McKeown

Heather Zhu(yz4042), Weiyao Xie (wx2251)

## Getting Started
### Software Requirement
- Python 3.7
- Numpy >= 1.18.1.
- sklearn 0.22.2
- transformers
- pandas 1.0.3
- scipy


### Usage
## Training:
Run word2vec_svd.py to train embeddings using word2vec and SVD.
Place the training corpus under data folder.
Models and matrixs will be saved into data folder accordingly.
```
python word2vec_svd.py
```

## Evaluation:
Word2vec and SVD: run evaluate.py to evaluate models on three different tasks. Need to use functions in process.py to run.
```
python evaluate.py data/word_matrix/W_10_300.txt
```

BERT: run bert_eval.py to generate and evaluate BERT embeddings on the same three tasks. 
The argument is optional. If proveided a txt file containing saved word embedding, BERT will run much faster on BATS. 
```
python bert_eval.py saved_embedding_file_path
```

## bert_eval.py: 
* Evaluate BERT embeddings on three tasks. 
* Return a dictionary containing all evaluation results. 

## word2vec_svd.py: 
* Compute 36 word embedding models and save them in the data folders.

## process.py: 
* Changed the load_model function to read word2vec model because we are using a different version of gensim. 




