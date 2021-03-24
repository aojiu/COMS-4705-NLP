import torch
from torch.utils import data
from scipy.stats import spearmanr
from transformers import BertTokenizer, BertModel
import numpy as np
from numpy import random
import os

import argparse

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cosine
from numpy.linalg import norm


from process import load_model



def collect(model):
    '''
        Collects matrix and vocabulary list from a trained model.
        Helper function. You shouldn't have to call this yourself.
    '''
    if type(model) is dict:
        vocab = [k for k in model.keys()]
    else:
        vocab = [k for k in model.vocab.keys()]

    indices = {}
    for i in range(len(vocab)): indices[vocab[i]] = i
        
    matrix = []
    for w in vocab:
        # add word embeddings in the model into the matrix
        matrix.append(model[w])

    return np.array(matrix), vocab, indices
    
# evaluate wordsim
def eval_wordsim_bert(bert, tokenizer, f='data/wordsim353/combined.tab'):
    '''
        Evaluates a trained embedding model on WordSim353 using cosine
        similarity and Spearman's rho. Returns a tuple containing
        (correlation, p-value).

        Read words from wordsim353 
    '''
#     f='wordsim353/combined.tab'
    sim = []
    pred = []
    file = open(f, 'r')
    next(file)
    for line in file:
        splits = line.split('\t')
        w1 = splits[0]
        w2 = splits[1]
        text=[w1,w2]
        pt_batch = tokenizer(text,
                     padding=True,
                     truncation=True,
                     max_length=3,
                     return_tensors="pt")
        output=bert(input_ids=pt_batch["input_ids"])
#         print(output.shape)
#         print(output.last_hidden_state.shape)

        # get the word embedding from BERT
        v1 = output.last_hidden_state[0,1,:].detach().numpy()

        v2 = output.last_hidden_state[1,1,:].detach().numpy()

        sim.append(float(splits[2]))
        pred.append(np.dot(v1, v2)/(np.linalg.norm(v1) * np.linalg.norm(v2)))
    return spearmanr(sim, pred)




# evaluate bats file
def eval_bats_file_bert(matrix, vocab, indices, f, bert, tokenizer,repeat=False, multi=0):
    '''
        Evaluates a trained embedding model on a single BATS file using either
        3CosAdd (the classic vector offset cosine method) or 3CosAvg (held-out
        averaging).

        If multi is set to zero or None, this function will usee 3CosAdd;
        otherwise it will use 3CosAvg, holding out (multi) samples at a time.

        Default behavior is to use 3CosAdd.
    '''

    pairs = [line.strip().split() for line in open(f, 'r').readlines()]

    # discard pairs that are not in our vocabulary
    pairs = [[p[0], p[1].split('/')] for p in pairs if p[0] in vocab]
    pairs = [[p[0], [w for w in p[1] if w in vocab]] for p in pairs]
    pairs = [p for p in pairs if len(p[1]) > 0]
    if len(pairs) <= 1: return None

    transposed = np.transpose(np.array([x / norm(x) for x in matrix]))
# BERT evaluation using 3CosAdd
    if not multi:
        qa = []
        qb = []
        qc = []
        targets = []
        exclude = []
        groups = []
        
        for i in range(len(pairs)):
            j = random.randint(0, len(pairs) - 2)
            if j >= i: j += 1
            w1=pairs[i][0]
            w2=pairs[j][0]
            # get seperate word embedding
            text=[w1,w2]
            pt_batch = tokenizer(text,
                     padding=True,
                     truncation=True,
                     max_length=3,
                     return_tensors="pt")
            output=bert(input_ids=pt_batch["input_ids"])
            
            a=output.last_hidden_state[0,1,:].detach().numpy()
            c=output.last_hidden_state[1,1,:].detach().numpy()

            for bw in pairs[i][1]:
                qa.append(a)
                text_b=[bw]
                pt_batch_b = tokenizer(text,
                     padding=True,
                     truncation=True,
                     max_length=3,
                     return_tensors="pt")
                output_b=bert(input_ids=pt_batch_b["input_ids"])
                out_bw=output_b.last_hidden_state[0,1,:].detach().numpy()
                qb.append(out_bw)
                qc.append(c)
                groups.append(i)
                targets.append(pairs[j][1])
                exclude.append([pairs[i][0], bw, pairs[j][0]])

        for queries in [qa, qb, qc]:
            queries = np.array([x / norm(x) for x in queries])
        
        sa = np.matmul(qa, transposed) + .0001
        sb = np.matmul(qb, transposed)
        sc = np.matmul(qc, transposed)
        sims = sb + sc - sa
        # exclude original query words from candidates
        for i in range(len(exclude)):
            for w in exclude[i]:

                sims[i][indices[w]] = 0

    # BERT evaluation using 3CosAvg
    else:
        offsets = []
        exclude = []
        preds = []
        targets = []
        groups = []
        
        for i in range(len(pairs) // multi):
            qa = [pairs[j][0] for j in range(len(pairs)) if j - i not in range(multi)]
            qb = [[w for w in pairs[j][1] if w in vocab] for j in range(len(pairs)) if j - i not in range(multi)]
            qbs = []
            for ws in qb: qbs += ws
            lst_a=[]
            for w in qa:
                pt_batch = tokenizer([w],
                     padding=True,
                     truncation=True,
                     max_length=3,
                     return_tensors="pt")
                output_a=bert(input_ids=pt_batch["input_ids"])
                out_a=output_a.last_hidden_state[0,1,:].detach().numpy()
                lst_a.append(out_a)
            a = np.mean(np.array(lst_a), axis=0)
#             a = np.mean([model[w] for w in qa], axis=0)
            lst_b=[]
            for w in ws:
                pt_batch = tokenizer([w],
                     padding=True,
                     truncation=True,
                     max_length=3,
                     return_tensors="pt")
                output_b=bert(input_ids=pt_batch["input_ids"])
                out_b=output_b.last_hidden_state[0,1,:].detach().numpy()
                lst_b.append(out_b)
            b = np.mean(np.array(lst_b), axis=0)
#             b = np.mean([np.mean([model[w] for w in ws], axis=0) for ws in qb], axis=0)
            a = a / norm(a)
            b = b / norm(b)

            for k in range(multi):
                text=pairs[i + k][0]
                pt_batch = tokenizer([text],
                     padding=True,
                     truncation=True,
                     max_length=3,
                     return_tensors="pt")
                output_c=bert(input_ids=pt_batch["input_ids"])
                c=output_c.last_hidden_state[0,1,:].detach().numpy()
#                 c = model[pairs[i + k][0]]
                c = c / norm(c)
                offset = b + c - a
                offsets.append(offset / norm(offset))
                targets.append(pairs[i + k][1])
                exclude.append(qa + qbs + [pairs[i + k][0]])
                groups.append(len(groups))

        print(np.shape(transposed))

        sims = np.matmul(np.array(offsets), transposed)
        print(np.shape(sims))
        for i in range(len(exclude)):
            for w in exclude[i]:
                sims[i][indices[w]] = 0
    preds = [vocab[np.argmax(x)] for x in sims]
    accs = [1 if preds[i].lower() in targets[i] else 0 for i in range(len(preds))]

    regrouped = np.zeros(np.max(groups) + 1)
    for a, g in zip(accs, groups):
        regrouped[g] = max(a, regrouped[g])
    return np.mean(regrouped)

def eval_bats_bert(wv, dict_keys, indices,bert, tokenizer):
    '''
        Evaluates a trained embedding model on BATS.

        Returns a dictionary containing
        { category : accuracy score over the category }, where "category" can
        be
            - any of the low-level category names (i.e. the prefix of any of
              the individual data files)
            - one of the four top-level categories ("inflectional_morphology",
              "derivational_morphology", "encyclopedic_semantics",
              "lexicographic_semantics")
            - "total", for the overall score on the entire corpus
    '''
    accs = {}
    base = 'data/BATS'
    # base = 'BATS'
    for dr in os.listdir('data/BATS'):
    # for dr in os.listdir('BATS'):
        if os.path.isdir(os.path.join(base, dr)):
            dk = dr.split('_', 1)[1].lower()
            accs[dk] = []
            for f in os.listdir(os.path.join(base, dr)):
                f='data/BATS/'+dr+"/"+f
                accs[f.split('.')[0]] =eval_bats_file_bert(wv, dict_keys, indices, f,bert, tokenizer)
                
                accs[dk].append(accs[f.split('.')[0]])
            accs[dk] = [a for a in accs[dk] if a is not None]
            accs[dk] = np.mean(accs[dk]) if len(accs[dk]) > 0 else None

    accs['total'] = np.mean([accs[k] for k in accs.keys() if accs[k] is not None])

    return accs
def load_msr(f, limit=None):
    '''
        Loads the MSR paraphrase corpus.
    '''
    lines = [x.strip().lower().split('\t') for x in open(f, 'r').readlines()[1:]]
#     sents = [[x[3].split(), x[4].split()] for x in lines]
    sents = [[x[3], x[4]] for x in lines]
    labels = [int(x[0]) for x in lines]
    return sents, labels


def get_lst_msr(sentence, bert, tokenizer, if_cls=0):
    pt_batch = tokenizer(
         [sentence],
         padding=True,
         truncation=True,
         max_length=512,
         return_tensors="pt")

    output=bert(input_ids=pt_batch["input_ids"])
    # just get the [CLS] sentence embedding
    if if_cls:
        wv1=output.last_hidden_state.detach().numpy()[:,0,:]
        wv1_final = wv1.reshape((-1,768))
    # get the sentence embedding from bert by summing all words embedding
    else:
        wv1=output.last_hidden_state.detach().numpy()
        wv1 = wv1.reshape((-1,768))
        wv1_final=np.sum(wv1,axis=0)
    return wv1_final

def eval_msr_bert(model,bert, tokenizer):
    '''
        Evaluates a trained embedding model on the MSR paraphrase task using
        logistic regression over cosine similarity scores.
    '''
    X_tr, y_tr = load_msr('data/msr/msr_paraphrase_train.txt')
    X_test, y_test = load_msr('data/msr/msr_paraphrase_test.txt')

    train= []
    test= []
    count=0

    for ss in X_tr:
        # for each sentence
        # get the sentence embedding from BERT [CLS]
        wv1_sum = get_lst_msr(ss[0], bert, tokenizer)
        wv2_sum = get_lst_msr(ss[1], bert, tokenizer)
        train.append([wv1_sum,wv2_sum])
        count+=1
        if count%5000==1:
            print(count)
            
#         print(np.array(train).shape)
    for ss in X_test:
        wv1_sum_test = get_lst_msr(ss[0], bert, tokenizer)
        wv2_sum_test = get_lst_msr(ss[1], bert, tokenizer)
        test.append([wv1_sum_test,wv2_sum_test])
#         print(np.array(test).shape)
    
    tr_cos = np.array([1 - cosine(x[0], x[1]) for x in train]).reshape(-1, 1)
    test_cos = np.array([1 - cosine(x[0], x[1]) for x in test]).reshape(-1, 1)
    print(tr_cos.shape)
    lr = LogisticRegression(class_weight='balanced', solver='liblinear')
    lr.fit(tr_cos, y_tr)
    preds = lr.predict(test_cos)

    return accuracy_score(y_test, preds)



def evaluate_models(files, bert_embedding=None,verbose=True):
    '''
        Evaluates multiple models at a time. Returns results in a list where
        each item is a dict containing
        { "wordsim" : WordSim353 correlation,
          "bats" : a dictionary of BATS scores (see eval_bats() for details),
          "msr" : MSR paraphrase performance }.
    '''
    print("evaluating bert")

    results_bert=[]
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert = BertModel.from_pretrained('bert-base-uncased')
    # get the vocab list
    model = load_model(files)
    matrix, vocab, indices = collect(model)
    pt_batch = tokenizer(vocab,padding=True,
                     truncation=True,
                     max_length=3,
                     return_tensors="pt")
    # get embedding for each word using bert for bats
    try:
        wv=np.loadtxt(bert_embedding)
        print("BERT embedding loaded")
    except:
        print("BERT embedding not given. Running BERT inference.")
        wv=np.array([])
        for elem in pt_batch["input_ids"]:
            if len(wv)==0:
                output=bert(input_ids=elem.view(1, 3))
                wv=output.last_hidden_state[:,1,:].detach().numpy()
            else:
                output_new=bert(input_ids=elem.view(1, 3))
                wv_new=output_new.last_hidden_state[:,1,:].detach().numpy()
                wv=np.vstack((wv,wv_new))
    # indices = {}
    # for i in range(len(dict_keys)): indices[dict_keys[i][0]] = i


    r_bert = {}
    if verbose: print('[evaluate_bert] Evaluating on WordSim...')
    r_bert['wordsim'] = eval_wordsim_bert(bert, tokenizer)
    if verbose: print('[evaluate_bert] Evaluating on BATS...')
    r_bert['bats'] = eval_bats_bert(wv, vocab, indices,bert, tokenizer)
    if verbose: print('[evaluate_bert] Evaluating on MSRPC...')
    r_bert['msr'] = eval_msr_bert(model, bert, tokenizer)
    results_bert.append(r_bert)

    # for f in files:
    #     if verbose: print('[evaluate_models] Reading ' + f)

    #     r = {}
    #     if verbose: print('[evaluate_models] Evaluating on WordSim...')
    #     r['wordsim'] = eval_wordsim(bert, tokenizer)
    #     if verbose: print('[evaluate_models] Evaluating on BATS...')
    #     r['bats'] = eval_bats(wv, dict_keys.reshape(len(dict_keys)), indices,bert, tokenizer)
    #     if verbose: print('[evaluate_models] Evaluating on MSRPC...')
    #     r['msr'] = eval_msr(model,bert, tokenizer)
    #     results.append(r)

        

    return results_bert

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Evaluate BERT embeddings')
    
    parser.add_argument('--embedding', metavar='embedding', type=str,required=False, help='the path to the file containing BERT embeddings')



    # we will use Brown Corpus for bert when evaluating BATS tasks
    # the first parameter into this function can be any word2vec model we saved
    # it just needs to get the vocab list from brown corpus 
    # so that it is fair to evaluate bert in BATS
    # The seconde parameter is optional
    # if given a saved embedding, it does not need to run bert on all sentences
    if args.embedding:
        results = evaluate_models("data/bert_eval/test.bin", args.embedding)
    else:
        results = evaluate_models("data/bert_eval/test.bin")
    print("The final evaluation results for BERT on three tasks: ")
    print(results)