import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from scipy.sparse import lil_matrix
from scipy.sparse import coo_matrix
from scipy.sparse import diags
from scipy.sparse import linalg
from scipy.sparse import save_npz
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from scipy.sparse.linalg import svds
import transformers
import torch
import logging
from transformers import AutoTokenizer, AutoModel

# read the brown corpus
df = pd.read_csv('brown.txt', delimiter = "\t",names=["text"],header=None)

df["text"]=df["text"].str.lower()
sentences=[word_tokenize(x) for x in df["text"]]


###  PMI+SVD helper functions


def calculate_pmi(coo):
    # input: co occurence matrix as coo_matrix 

    # convert coo matrix to lil matrix
    arr = lil_matrix(coo)
    
    # calculate the sum of p(y|x) probability within the row
    row_totals = lil_matrix.sum(arr, axis=1)
    prob_cols_given_row = lil_matrix.transpose(lil_matrix.transpose(arr).multiply(1 / row_totals))

    # calculate the sum of p(y) probability within the column
    col_totals = lil_matrix.sum(arr,axis=0)
    prob_of_cols = col_totals / lil_matrix.sum(col_totals)


    # calculate PMI: log( p(y|x) / p(y) )
    ratio = prob_cols_given_row / prob_of_cols

    ratio[ratio<1] = 1
    # pmi=lil_matrix((np.log(ratio).shape[0],np.log(ratio).shape[1]))
    
    pmi= lil_matrix(np.log(ratio))

    return pmi


def find_distinct_words(corpus):
    """ Determine a list of distinct words for the corpus.
        Params:
            corpus (list of list of strings): corpus of documents
        Return:
            corpus_words (list of strings): list of distinct words across the corpus, sorted (using python 'sorted' function)
            num_corpus_words (integer): number of distinct words across the corpus
    """
    corpus_words = []
    num_corpus_words = -1
    
    corpus_words = [word for line in corpus for word in line]
    corpus_words_set = set(corpus_words)
    corpus_words = sorted(list(corpus_words_set))
    num_corpus_words = len(corpus_words)


    return corpus_words, num_corpus_words


def compute_co_occurrence_matrix(corpus, window_size):
    """ Compute co-occurrence matrix for the given corpus and window_size.

        Return:
            M (lil_matrix of shape (number of corpus words, number of corpus words)): 
                Co-occurence matrix of word counts. 
                The ordering of the words in the rows/columns should be the same as the ordering of the words given by the distinct_words function.
            word2Ind (dict): dictionary that maps word to index (i.e. row/column number) for matrix M.
    """
    words, num_words = find_distinct_words(corpus)
    M = None
    word2Ind = {}
    

    for i in range(num_words):
        word2Ind[words[i]] = i
    M = lil_matrix((num_words, num_words))
    for line in corpus:
        for i in range(len(line)):
            target = line[i]
            target_index = word2Ind[target]
            
            left = max(i - window_size, 0)
            right = min(i + window_size, len(line) - 1)
            for j in range(left, i):
                window_word = line[j]
                M[target_index, word2Ind[window_word]] += 1
                M[word2Ind[window_word], target_index] += 1

    # save M as coo_matrix
    save_npz('matrix/cooccur_{}'.format(window_size),coo_matrix(M))
    # return M as lil_matrix
    return M, word2Ind


# compute svd using scipy sparse svd
# using numpy svd will take very long time
def get_wc_sparse(pmiM, K):
    U, s, VT = svds(pmiM, K)
    s = diags(s)
    s_sq = np.sqrt(s)
    W = U@s_sq
    C= VT.T @ s_sq
    return W,C




# save models for 36 settings of parameters

context_window_size_lst = [2, 5, 10]
dimension_lst = [50, 100,300]
num_negsamp_lst = [1, 5, 15]

for context_window_size in context_window_size_lst:
    # get cooccurrence with different window size
    # get the vocab list
    M,ind_dict = compute_co_occurrence_matrix(sentences, window_size=context_window_size)
    dict_keys = np.array(list(ind_dict.keys()))[:, np.newaxis]

    # get pmi matrix for each window size and dimension
    for dimension in dimension_lst:
        pmi_mat=calculate_pmi(M)
        # get w, c with top k dimension
        W,C=get_wc_sparse(pmi_mat, dimension)
        word_mat = np.concatenate((dict_keys, W), axis=1)
        print(word_mat.shape)
        word_mat = word_mat.astype(str)
        context_mat = np.concatenate((dict_keys, C), axis=1).reshape(-1,dimension+1)
        context_mat = context_mat.astype(str)
        np.savetxt("word_matrix/W_{}_{}.txt".format(context_window_size,dimension), word_mat, fmt="%s")
        np.savetxt("context_matrix/C_{}_{}.txt".format(context_window_size,dimension), context_mat, fmt="%s")
        
        for num_negsmap in num_negsamp_lst:
            model = Word2Vec(sentences=sentences, size=dimension, 
                             window=context_window_size, negative=num_negsmap, min_count=1, workers=4)
            model.wv.save("word_vec/word2vec{}_{}_{}.bin".format(context_window_size, dimension, num_negsmap))
            