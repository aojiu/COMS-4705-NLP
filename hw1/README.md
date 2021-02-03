# Homework 1: Supervised Text Classification
Natural Language Processing
Weiyao Xie (wx2251)

## Getting Started
### Software Requirement
- Python 3.7
- Numpy >= 1.18.1.
- sklearn 0.22.2
- NLTK 3.5
- pandas 1.0.3



### Usage
Call hw1.py to train the selected model and get the test results.
There are four required arguments to pass in.
```
python hw1.py --train <train data> --test <test data> --model <model name> --lexicon_path <lexicon_path>
```

For assignment1, input image is stored at ```<image_analysis dir>/assignment1/Homework1```. The output image is stored at ```<image_analysis dir>/assignment1/output```.

## hw1.py: 
* This is the main function.
* First, read training and test data.
* Preprocess texts by converting everything to lower letter; converting all urls to ||url||.
* Extract ngram features by calling functions in features.py
* Use ```SelectKBest``` to reduce number of features to 1000 so that training time will be shorter
```
feature_select=SelectKBest(f_classif, k=1000)
sub_training_X = feature_select.fit_transform(sub_training_X, training_Y)
sub_test_X = feature_select.transform(sub_test_X)
```
* Extract lexicon features by calling functions in features.py
* Extract lexicon features by calling functions in features.py(Used number of capital letters and number of hashtags)
* Extract additional features(used POS tags and number of exclamation mark)
* Perform Standardization on all features except for ngram features so that sparsity can be preserved. 
```
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
```
* Initialize linear SVM with stochastic gradient descent optimization
```
sgd = SGDClassifier(loss="hinge",penalty="elasticnet", l1_ratio=0.05, random_state=43, max_iter=6000)
```
* Train/test the model and report final macro f1 score and f1 score for each class


## features.py: 
* The file contains all functions called in hw1.py
* In ngram_feature, uses TfidfVectorizer to get ngram features
```
word_ngram_1 = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words="english", ngram_range=(1,4))
```
* In ngram_feature, uses TfidfVectorizer to get char ngram features by changing ```analyzer= "char"```
```
char_ngram_1 = TfidfVectorizer(sublinear_tf=True, analyzer="char",max_df=0.6, stop_words="english", ngram_range=(3,5))
```
* When extracing lexicon features, uses NLTK's TweeterTokenizer to better process tweets
```
tknzr = TweetTokenizer()
token_lst = tknzr.tokenize(tweet)
```
  
## Limitations/Future Improvements
* Instead of using SVC in sklearn, I used SGDClassifier with hinge loss which is essentially linear SVM. When it is much faster to converge, I could not use rbf or polynomial kernel in SVC. Since I have more than 1000 features in my final training data, it is too time consuming to use SVC. I think in the future I can try to reduce some ngram features. When the number of features is lower, it will be more efficient and more likely to get a better result by trying different SVM kernels.
* For the best K ngram features I selected, I did not do a very thorough fine tuning. I think the model can perform well even with less features. 



