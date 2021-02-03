from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from nltk.tokenize import TweetTokenizer
import csv

################### Text Preprocessing ###################
"""
use df["tweet_tokens"].apply() to call the function
"""
def preprocessing_url(string):
    # replace the url with tag ||url||
    return re.sub(r'(http\S+)|(https\S+)', '||url||', string)

def preprocessing_upper_lower(string):
    # this function should be applied after counting nunmber of upper cases
    # but before ngram
    return string.lower()

def len_tweet(string):
    tknzr = TweetTokenizer()
    token_lst = tknzr.tokenize(string)
    return len(token_lst)

def preprocessing_username(string):
    # replace user name starting with @ with ||user||
    return re.sub(r'@\S+', '||user||', string)

def strip(string):
    return string.replace(" ", "")

def vectorcount(ngram, train_data_token, test_data_token):
    # word_ngram_1 = CountVectorizer(analyzer='word', ngram_range=(int(ngram),int(ngram)))
    word_ngram_1 = TfidfVectorizer(sublinear_tf=True, max_df=0.6, stop_words="english", ngram_range=(int(ngram),int(ngram)))
    train_word_ngram_feat = word_ngram_1.fit_transform(train_data_token)
    test_word_ngram_feat = word_ngram_1.transform(test_data_token)
    
    train_word_ngram_feat = np.array(train_word_ngram_feat.toarray())
    test_word_ngram_feat = np.array(test_word_ngram_feat.toarray())
    return train_word_ngram_feat, test_word_ngram_feat

################### Ngram Features ###################
def ngram_feature(df_train, df_test):
    """
    input:
    data_token: list of texts

    output:
    ngram_feat: sparse matrix that contains ngram occurances
    """
    # read train/test data
    train_str_label = df_train["label"].values
    test_str_label = df_test["label"].values
    
    # convert string labels to int
    # d = dict([(y,x) for x,y in enumerate(set(train_str_label))])
    d={"negative":0, "positive":1, "neutral":2}
    train_labels = [d[x] for x in train_str_label]
    test_labels = [d[x] for x in test_str_label]
    
    #seperate text data from labels
    train_data_token=df_train["tweet_tokens"].values
    test_data_token=df_test["tweet_tokens"].values
     

    # get normalized char ngram from 
    word_ngram_1 = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words="english", ngram_range=(1,4))
    train_word_ngram_feat = word_ngram_1.fit_transform(train_data_token)
    test_word_ngram_feat = word_ngram_1.transform(test_data_token)

    train_ngram_feat = np.array(train_word_ngram_feat.toarray())
    test_ngram_feat = np.array(test_word_ngram_feat.toarray())
    
    print(train_ngram_feat.shape)
    print(test_ngram_feat.shape)
    print(len(train_labels))
    print(len(test_labels))
    return train_ngram_feat, train_labels, test_ngram_feat, test_labels

################### Char Features ####################
def char_feature(df_train, df_test):
    """
    input:
    data_token: list of texts

    output:
    ngram_feat: sparse matrix that contains ngram occurances
    """
    
    #seperate text data from labels
    train_data_token=df_train["tweet_tokens"].values
    test_data_token=df_test["tweet_tokens"].values
     
    
    # get normalized ngram from 1-4

    char_ngram_1 = TfidfVectorizer(sublinear_tf=True, analyzer="char",max_df=0.6, stop_words="english", ngram_range=(3,5))
    train_word_ngram_feat = char_ngram_1.fit_transform(train_data_token)
    test_word_ngram_feat = char_ngram_1.transform(test_data_token)
    
    train_word_ngram_feat = np.array(train_word_ngram_feat.toarray())
    test_word_ngram_feat = np.array(test_word_ngram_feat.toarray())

    return train_word_ngram_feat, test_word_ngram_feat




###################### Lexicon Features #######################
from nltk.tokenize import TweetTokenizer
def unigram_lex(df, lex_loc):
    lexicon_uni=pd.read_csv(lex_loc, sep="\t", header=None, names=["token", "score", "pos_num", "neg_num"])
    token_array = df["tweet_tokens"].values
    map_dict = pd.Series(lexicon_uni.score.values,index=lexicon_uni.token.values).to_dict()
#     print(map_dict)
    uni_score_pos = []
    uni_count_pos = []
    uni_max_score=[]
    uni_last_pos=[]
    
    uni_score_neg = []
    uni_count_neg = []
    uni_min_score=[]
    uni_last_neg=[]
    total_score=[]

    for tweet in token_array:
        tknzr = TweetTokenizer()
        token_lst = tknzr.tokenize(tweet)

        # get sum of score larger than 0
        score_for_tweet_pos = 0
        score_for_tweet_neg = 0
        count_pos=0
        count_neg=0
        max_score=-2**31
        min_score=2**31
        last_pos=0
        last_neg=0
        total_score_tweet=0
        for token in token_lst:
            try:
                token_lex_score = map_dict[token]
            except:
                token_lex_score=0
#                 print(token_lex_score)
            total_score_tweet+=token_lex_score
            # get count of tokens that have scores higher than 0
            if float(token_lex_score)>0:
                count_pos+=1
                last_pos=token_lex_score
                score_for_tweet_pos+=float(token_lex_score)
            elif float(token_lex_score)<0:
                count_neg+=1
                last_neg=token_lex_score
                score_for_tweet_neg+=float(token_lex_score)
            
            # get max score of tweets
            if float(token_lex_score)>max_score:
                max_score=float(token_lex_score)
            # get min score of tweets 
            if float(token_lex_score)<min_score:
                    min_score=float(token_lex_score)

        # positive features
        uni_score_pos.append(score_for_tweet_pos)
        uni_count_pos.append(count_pos)
        uni_max_score.append(max_score)
        uni_last_pos.append(last_pos)
        total_score.append(total_score_tweet)
        # negtive features
        uni_score_neg.append(score_for_tweet_neg)
        uni_count_neg.append(count_neg)
        uni_min_score.append(min_score)
        uni_last_neg.append(last_neg)
        
    uni_score_pos = np.array(uni_score_pos).reshape(len(uni_score_pos),-1)
    uni_count_pos = np.array(uni_count_pos).reshape(len(uni_count_pos),-1)
    uni_max_score = np.array(uni_max_score).reshape(len(uni_max_score),-1)
    uni_last_pos = np.array(uni_last_pos).reshape(len(uni_last_pos),-1)
    
    uni_score_neg = np.array(uni_score_neg).reshape(len(uni_score_neg),-1)
    uni_count_neg = np.array(uni_count_neg).reshape(len(uni_count_neg),-1)
    uni_min_score = np.array(uni_min_score).reshape(len(uni_min_score),-1)
    uni_last_neg = np.array(uni_last_neg).reshape(len(uni_last_neg),-1)
    
    total_score = np.array(total_score).reshape(len(total_score),-1)
    
    final = np.concatenate((uni_score_pos, uni_count_pos), axis=1)
    final=np.concatenate((final, uni_max_score), axis=1)
    final=np.concatenate((final, uni_last_pos), axis=1)
    # final=np.concatenate((final, total_score), axis=1)
    
    final=np.concatenate((final, uni_score_neg), axis=1)
    final=np.concatenate((final, uni_count_neg), axis=1)
    final=np.concatenate((final, uni_min_score), axis=1)
    final=np.concatenate((final, uni_last_neg), axis=1)
    
    return final, total_score





def bigram_lex(df, lex_loc):
    
    lexicon_bi=pd.read_csv(lex_loc, sep="\t", header=None, encoding="utf8", quoting=csv.QUOTE_NONE,names=["token", "score", "pos_num", "neg_num"])
    token_array = df["tweet_tokens"].values
    bi_score_pos = []
    bi_count_pos = []
    bi_max_score=[]
    bi_last_pos=[]
    
    total_score=[]
    
    bi_score_neg = []
    bi_count_neg = []
    bi_min_score=[]
    bi_last_neg=[]
    
    total_count=0
    map_dict = pd.Series(lexicon_bi.score.values,index=lexicon_bi.token.values).to_dict()

    for tweet in token_array:
        total_count+=1
        tknzr = TweetTokenizer()
        token_lst = tknzr.tokenize(tweet)

        # get sum of score larger than 0
        score_for_tweet_pos = 0
        score_for_tweet_neg = 0
        count_pos=0
        count_neg=0
        total_score_tweet=0
        max_score=-2**31
        min_score=2**31
        last_pos=0
        last_neg=0
        for i in range(len(token_lst)-1): 
            bi_str = token_lst[i]+" "+token_lst[i+1]
            try:
                token_lex_score = map_dict[bi_str]
            except:
                token_lex_score=0
            total_score_tweet+=token_lex_score
            if float(token_lex_score)>=0:
                score_for_tweet_pos+=float(token_lex_score)
            else:
                score_for_tweet_neg+=float(token_lex_score)
            # get count of tokens that have scores higher than 0
            if float(token_lex_score) >= 0:
                count_pos+=1
                last_pos=token_lex_score
            else:
                count_neg+=1
                last_neg=token_lex_score
                    
                # get max score of tweets
            if float(token_lex_score)>max_score:
                max_score=float(token_lex_score)
            if float(token_lex_score)<min_score:
                min_score=float(token_lex_score)
            
        
                
    #
        bi_score_pos.append(score_for_tweet_pos)
        bi_count_pos.append(count_pos)
        bi_max_score.append(max_score)
        bi_last_pos.append(last_pos)
        
        bi_score_neg.append(score_for_tweet_neg)
        bi_count_neg.append(count_neg)
        bi_min_score.append(min_score)
        bi_last_neg.append(last_neg)
        
        total_score.append(total_score_tweet)
        
    bi_score_pos = np.array(bi_score_pos).reshape(len(bi_score_pos),-1)
    bi_count_pos = np.array(bi_count_pos).reshape(len(bi_count_pos),-1)
    bi_max_score = np.array(bi_max_score).reshape(len(bi_max_score),-1)
    bi_last_pos = np.array(bi_last_pos).reshape(len(bi_last_pos),-1)
    
    total_score = np.array(total_score).reshape(len(total_score),-1)
    
    bi_score_neg = np.array(bi_score_neg).reshape(len(bi_score_neg),-1)
    bi_count_neg = np.array(bi_count_neg).reshape(len(bi_count_neg),-1)
    bi_min_score = np.array(bi_min_score).reshape(len(bi_min_score),-1)
    bi_last_neg = np.array(bi_last_neg).reshape(len(bi_last_neg),-1)
    
    final = np.concatenate((bi_score_pos, bi_count_pos), axis=1)
    final=np.concatenate((final, bi_max_score), axis=1)
    final=np.concatenate((final, bi_last_pos), axis=1)
    
    # final=np.concatenate((final, total_score), axis=1)
    
    final=np.concatenate((final, bi_score_neg), axis=1)
    final=np.concatenate((final, bi_count_neg), axis=1)
    final=np.concatenate((final, bi_min_score), axis=1)
    final=np.concatenate((final, bi_last_neg), axis=1)
    return final, total_score






################### Encoding-based features ###################

import re
# number of caps has better improvement on performance than number of all caps strings
def get_caps(string):
    return sum([1 for c in string if c.isupper()])

def hashtags(string):
    return sum([1 for c in string if c=="#"])

def pos_words(string):
    return sum([1 for c in string if c in ["best", "great", "good", "thanks", "like", "love", "amazing", "cool"]])

def exclaim(string):
    return sum([1 for c in string if c=="!"])

def question(string):
    return sum([1 for c in string if c=="?"])

def user_num(string):
    return sum([1 for c in string if c=="@"])

def strip(string):
    return string.replace(" ", "")

def pos_occurence(df_train, df_test):
    char_ngram_train = CountVectorizer(analyzer='char', ngram_range=(1,1))
    
    train_pos_occurance = char_ngram_train.fit_transform(df_train["pos_tags"])
    test_pos_occurance = char_ngram_train.transform(df_test["pos_tags"])

    train_pos_occurance = np.array(train_pos_occurance.toarray())
    test_pos_occurance = np.array(test_pos_occurance.toarray())

    return train_pos_occurance, test_pos_occurance


def num_elongated(string):
    regex = re.compile(r"(.)\1{2}")
    return len([word for word in string.split() if regex.search(word)])

if __name__ == '__main__':

    ngram_feature("data/dev.csv")
