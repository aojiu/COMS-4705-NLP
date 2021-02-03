import argparse
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import SelectKBest, chi2, f_classif
import numpy as np
from sklearn.model_selection import GridSearchCV
from features import ngram_feature
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from features import preprocessing_upper_lower
from features import preprocessing_url
from features import preprocessing_username
from features import strip
from features import hashtags
from features import get_caps
from features import pos_occurence
from features import num_elongated
from features import unigram_lex
from features import bigram_lex
from features import exclaim
from features import user_num
from features import char_feature
from features import question
from features import pos_words
from features import len_tweet
from sklearn import preprocessing
import csv

def model_training_testing(train_data_file, test_data_file, model_name, lexicon_path):
    # read data
    df_train = pd.read_csv(train_data_file)
    df_test = pd.read_csv(test_data_file)

    # change label
    df_train["label"]=df_train["label"].replace("objective", "neutral")
    df_test["label"]=df_test["label"].replace("objective", "neutral")

    # data preprocessing
    # get number of caps before converting everything to lower case
    train_caps = np.array(df_train["tweet_tokens"].apply(get_caps)).reshape(df_train.shape[0],-1)
    test_caps = np.array(df_test["tweet_tokens"].apply(get_caps)).reshape(df_test.shape[0],-1)

    df_train["tweet_tokens"] = df_train["tweet_tokens"].apply(preprocessing_upper_lower).values
    df_test["tweet_tokens"] = df_test["tweet_tokens"].apply(preprocessing_upper_lower).values
    
    df_train["tweet_tokens"] = df_train["tweet_tokens"].apply(preprocessing_url).values
    df_test["tweet_tokens"] = df_test["tweet_tokens"].apply(preprocessing_url).values

    df_train["pos_tags"]=df_train["pos_tags"].fillna(value=" ")
    df_test["pos_tags"]=df_test["pos_tags"].fillna(value=" ")

    df_train["pos_tags"] = df_train["pos_tags"].apply(strip)
    df_test["pos_tags"] = df_test["pos_tags"].apply(strip)

    # prepare ngram word data
    sub_training_X, training_Y, sub_test_X, test_Y  = ngram_feature(df_train,df_test)
    # prepare ngram char data
    char_training_X, char_test_X = char_feature(df_train,df_test)
    # concatenate them together
    sub_training_X = np.concatenate((sub_training_X, char_training_X), axis=1)
    sub_test_X = np.concatenate((sub_test_X, char_test_X), axis=1)
    print("ngram done")

    # create model based on arguement
    feature_select=SelectKBest(f_classif, k=1000)
    sub_training_X = feature_select.fit_transform(sub_training_X, training_Y)
    sub_test_X = feature_select.transform(sub_test_X)

    # lexicon model
    if (model_name=="Ngram+Lex") or (model_name =="Ngram+Lex+Enc") or (model_name =="Custom"):
        # get lexicon features
        lex_emo_uni = lexicon_path + "/Sentiment140-Lexicon/Emoticon-unigrams.txt"
        lex_emo_bi = lexicon_path +"/Sentiment140-Lexicon/Emoticon-bigrams.txt"

        lex_hs_uni = lexicon_path +"/Hashtag-Sentiment-Lexicon/HS-unigrams.txt"
        lex_hs_bi = lexicon_path +"/Hashtag-Sentiment-Lexicon/HS-bigrams.txt"


        uni_train, uni_train_total = unigram_lex(df_train, lex_emo_uni)
        uni_test, uni_test_total = unigram_lex(df_test, lex_emo_uni)

        bi_train, bi_train_total= bigram_lex(df_train, lex_emo_bi)
        bi_test, bi_test_total = bigram_lex(df_test, lex_emo_bi)

        hs_uni_train, hs_uni_train_total = unigram_lex(df_train, lex_hs_uni)
        hs_uni_test, hs_uni_test_total = unigram_lex(df_test, lex_hs_uni)

        hs_bi_train, hs_bi_train_total= bigram_lex(df_train, lex_hs_bi)
        hs_bi_test, hs_bi_test_total = bigram_lex(df_test, lex_hs_bi)

        # combine lexicon features with ngram features 
        # training_X = np.concatenate((training_X, uni_train), axis=1)
        training_X = uni_train
        test_X = uni_test
        
        training_X = np.concatenate((training_X, bi_train), axis=1)
        test_X = np.concatenate((test_X, bi_test), axis=1)

        training_X = np.concatenate((training_X, hs_uni_train), axis=1)
        test_X = np.concatenate((test_X, hs_uni_test), axis=1)
        training_X = np.concatenate((training_X, hs_bi_train), axis=1)
        test_X = np.concatenate((test_X, hs_bi_test), axis=1)

        # add total score in lexicon feature
        training_X = np.concatenate((training_X, uni_train_total), axis=1)
        test_X = np.concatenate((test_X, uni_test_total), axis=1)
        training_X = np.concatenate((training_X, bi_train_total), axis=1)
        test_X = np.concatenate((test_X, bi_test_total), axis=1)

        training_X = np.concatenate((training_X, hs_uni_train_total), axis=1)
        test_X = np.concatenate((test_X, hs_uni_test_total), axis=1)
        training_X = np.concatenate((training_X, hs_bi_train_total), axis=1)
        test_X = np.concatenate((test_X, hs_bi_test_total), axis=1)
        print("Lex done")
    
    if model_name =="Ngram+Lex+Enc":
        train_encoding = np.array(df_train["tweet_tokens"].apply(hashtags)).reshape(df_train.shape[0],-1)
        train_encoding = np.concatenate((train_encoding,train_caps), axis = 1)
        train_encoding = np.concatenate((train_encoding,np.array(df_train["tweet_tokens"].apply(exclaim)).reshape(df_train.shape[0],-1)), axis = 1)
        
        test_encoding = np.array(df_test["tweet_tokens"].apply(hashtags)).reshape(df_test.shape[0],-1)
        test_encoding = np.concatenate((test_encoding,test_caps), axis = 1)
        test_encoding = np.concatenate((test_encoding,np.array(df_test["tweet_tokens"].apply(exclaim)).reshape(df_test.shape[0],-1)), axis = 1)

        train_pos, test_pos = pos_occurence(df_train, df_test)
        training_X = np.concatenate((training_X, train_encoding), axis=1)
        test_X = np.concatenate((test_X, test_encoding), axis=1)
        
        training_X = np.concatenate((training_X, train_pos), axis=1)
        test_X = np.concatenate((test_X, test_pos), axis=1)
        print("Enc done")


    

        

    if model_name =="Custom":
        # previous encoding
        train_encoding = np.array(df_train["tweet_tokens"].apply(hashtags)).reshape(df_train.shape[0],-1)
        train_encoding = np.concatenate((train_encoding,train_caps), axis = 1)

        # additional features
        train_encoding = np.concatenate((train_encoding,np.array(df_train["tweet_tokens"].apply(exclaim)).reshape(df_train.shape[0],-1)), axis = 1)
        
        # previous encoding
        test_encoding = np.array(df_test["tweet_tokens"].apply(hashtags)).reshape(df_test.shape[0],-1)
        test_encoding = np.concatenate((test_encoding,test_caps), axis = 1)

        # additional features
        test_encoding = np.concatenate((test_encoding,np.array(df_test["tweet_tokens"].apply(exclaim)).reshape(df_test.shape[0],-1)), axis = 1)

        
        train_pos, test_pos = pos_occurence(df_train, df_test)

        training_X = np.concatenate((training_X, train_encoding), axis=1)
        test_X = np.concatenate((test_X, test_encoding), axis=1)
        
        training_X = np.concatenate((training_X, train_pos), axis=1)
        test_X = np.concatenate((test_X, test_pos), axis=1)
        print("Custom done")
        


    if model_name != "Ngram":
    # scale data to have 0 mean and unit variance
        scaler = preprocessing.StandardScaler().fit(training_X)
        training_X = scaler.transform(training_X)
        test_X=scaler.transform(test_X)
        training_X = np.concatenate((sub_training_X, training_X), axis=1)
        test_X = np.concatenate((sub_test_X, test_X), axis=1)
    else:
        training_X = sub_training_X
        test_X=sub_test_X


    # # SGD
    sgd = SGDClassifier(loss="hinge",penalty="elasticnet", l1_ratio=0.05, random_state=43, max_iter=6000)
    print("training")
    sgd.fit(training_X, training_Y)
    print("testing")
    predictions = sgd.predict(test_X)
    f1 = f1_score(test_Y, predictions, average='macro')
    class_score = f1_score(test_Y, predictions, average=None)
    print(class_score)
    print(f1)


    
    # naive bayes classifier
    # Gaussian 
    print("Naive Bayes training")
    gnb = GaussianNB()
    print("training")
    gnb.fit(training_X,training_Y)
    predictions_train = gnb.predict(training_X)
    f1_train = f1_score(training_Y, predictions_train, average='macro')
    print("training macro f1 score is {}".format(f1_train))
    print("testing")
    predictions = gnb.predict(test_X)
    f1 = f1_score(test_Y, predictions, average='macro')
    print("macro f1 score is {}".format(f1))

    # naive bayes classifier
    # Multinomial
    training_X=np.absolute(training_X)
    test_X=np.absolute(test_X)
    print("Multinomial Bayes training")
    gnb = MultinomialNB(alpha=0.1)
    print("training")
    gnb.fit(training_X,training_Y)
    predictions_train = gnb.predict(training_X)
    f1_train = f1_score(training_Y, predictions_train, average='macro')
    print("training macro f1 score is {}".format(f1_train))
    print("testing")
    predictions = gnb.predict(test_X)
    f1 = f1_score(test_Y, predictions, average='macro')
    print("macro f1 score is {}".format(f1))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', dest='train', required=True,
                        help='Full path to the training file')
    parser.add_argument('--test', dest='test', required=True,
                        help='Full path to the evaluation file')
    parser.add_argument('--model', dest='model', required=True,
                        choices=["Ngram", "Ngram+Lex", "Ngram+Lex+Enc", "Custom"],
                        help='The name of the model to train and evaluate.')
    parser.add_argument('--lexicon_path', dest='lexicon_path', required=True,
                        help='The full path to the directory containing the lexica.'
                             ' The last folder of this path should be "lexica".')
    args = parser.parse_args()

    model_training_testing(args.train, args.test, args.model, args.lexicon_path)