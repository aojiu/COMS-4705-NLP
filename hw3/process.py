'''
Data processing helper functions.
'''

import nltk
from nltk.corpus import brown
from nltk.tokenize import casual

import json

from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import numpy as np

# added
from itertools import islice

def load_msr(f, limit=None):
    '''
        Loads the MSR paraphrase corpus.
    '''
    # lines = [x.strip().lower().split('\t') for x in open(f, 'r').readlines()[1:]]
    lines = [x.strip().lower().split('\t') for x in open(f, 'r', encoding='utf8').readlines()[1:]]
    sents = [[x[3].split(), x[4].split()] for x in lines]
    labels = [int(x[0]) for x in lines]
    return sents, labels

def load_w2v(f):
    '''
        A wrapper for loading with gensim's KeyedVectors in word2vec format.
    '''
    return KeyedVectors.load_word2vec_format(f, binary=f.endswith('.bin'))
    # return KeyedVectors.load_word2vec_format(f, binary=f.endswith('.bin'),unicode_errors='ignore', encoding = 'windows-1252')


def load_kv(f):
    '''
        A wrapper for loading with gensim's KeyedVectors.
    '''

    return KeyedVectors.load(f)
    
    # ld = KeyedVectors.load(f)
    # print(len(ld.wv.vocab))
    # return ld


# # added

# def take(n, iterable):
#     "Return first n items of the iterable as a list"
#     return list(islice(iterable, n))

# # ended added

def load_txt(f):
    '''
        Loads vectors from a text file.
    '''
    vectors = {}

    # counter = float(0)
    
    for line in open(f, 'r').readlines():
        splits = line.strip().split()
        # vectors[splits[0]] = np.array([float(x) for x in splits[1:]])
        vectors[splits[0]] = np.array([0.00024177048424621904 if x =='0.00024177048424621904\x00\x00\x00\x00\x00\x02' else float(x) for x in splits[1:]])

    #     counter += sum(vectors[splits[0]])
    # # print(take(2, vectors.items()))
    # print("sum")
    # print(counter)

    return vectors



    # if f.endswith('.bin'):
    #     r['msr'] = eval_msr(model)
    # elif f.endswith('.txt'):
    #     print(11234325)
    #     dct = {}
    #     lines = f.split('\n')[:-1]
    #     for i in range(len(lines)):
    #         splits = lines[i].split(" ", 1)
    #         word = splits[0].strip()
    #         vector = splits[1].strip().split(' ')
    #         vector = list(map(float, vector))
    #         # vector = str(' [') + splits[1].strip() + str('] \n')
    #         dct[word] = vector
    #     r['msr'] = eval_msr(dct)
    # return vectors

def load_model(f):
    '''
        Guesses the file format and loads.
    '''
    # if f.endswith('.bin'): return load_w2v(f)
    if f.endswith('.bin'): return load_kv(f)
    elif f.endswith('.txt'): return load_txt(f)
    else: return load_kv(f)
