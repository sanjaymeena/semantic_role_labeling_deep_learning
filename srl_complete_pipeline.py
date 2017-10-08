# -*- coding: utf-8 -*-
import math
import numpy as np
import gzip
import paddle.v2 as paddle
import paddle.v2.evaluator as evaluator 
from  paddle.v2.parameters import  Parameters 
import srl_data_feeder
import pi_data_feeder

from paddle.trainer_config_helpers import *
from  srl_db_lstm import db_lstm
from pi_net import predicate_identifier_net
import math

# srl model 
srl_model_file="models/parameters/final/srl_params_pass_600.tar.gz"
#predicate identifier model
pi_model_file='models/final/srl_pi_params_pass_300.tar.gz'

# get data dicts for predicate identifier (PI)
pi_word_dict, pi_label_dict = pi_data_feeder.get_dict()
# get data dict for srl 
srl_word_dict, srl_verb_dict, srl_label_dict = srl_data_feeder.get_dict()
    
UNK_IDX=pi_data_feeder.UNK_IDX


# Load the predicate identifier model 
def loadPredicateIdentifierModel(model_file=pi_model_file):

    word_dict_len = len(pi_word_dict)
    label_dict_len = len(pi_label_dict)
    temp=109
    predict = predicate_identifier_net(word_dict_len,label_dict_len,is_train=False)
    print 'extracting  predicate identifier(PI) parameters from : ', model_file ,' ...'
    parameters=paddle.parameters.Parameters.from_tar(gzip.open(model_file))

    print 'initializing (PI) model ..'
    pi_inferer = paddle.inference.Inference(
                     output_layer=predict, parameters=parameters)
    print 'done loading the (PI) model  model.'

    return pi_inferer


# load the srl db lstm model 
def loadSRLModel(model_file=srl_model_file):
    
    word_dict_len = len(srl_word_dict)
    label_dict_len = len(srl_label_dict)
    pred_len = len(srl_verb_dict)

    feature_out=db_lstm(word_dict_len,label_dict_len,pred_len)
    print 'extracting  srl db-lstm model parameters from : ', model_file ,' ...'
    with gzip.open(model_file, 'r') as f:
        parameters = Parameters.from_tar(f)


    predict = paddle.layer.crf_decoding(
        size=label_dict_len,
        input=feature_out,
        param_attr=paddle.attr.Param(name='crfw'))
    
    print 'initializing srl db-lstm model ..'
    srl_inferer = paddle.inference.Inference(
                 output_layer=predict, parameters=parameters)
    print 'done loading the srl db-lstm  model.'

    return srl_inferer


def featureGenHelper(sentence,labels):
    sen_len = len(sentence)
    
    if 'B-V' not in labels:
        print 'B-V not present : ', labels,word_idx
    
    verb_index = labels.index('B-V')
    predicate=sentence[verb_index]

    mark = [0] * len(labels)
    if verb_index > 0:
        mark[verb_index - 1] = 1
        ctx_n1 = sentence[verb_index - 1]
    else:
        ctx_n1 = 'bos'

    if verb_index > 1:
        mark[verb_index - 2] = 1
        ctx_n2 = sentence[verb_index - 2]
    else:
        ctx_n2 = 'bos'

    mark[verb_index] = 1
    ctx_0 = sentence[verb_index]

    if verb_index < len(labels) - 1:
        mark[verb_index + 1] = 1
        ctx_p1 = sentence[verb_index + 1]
    else:
        ctx_p1 = 'eos'

    if verb_index < len(labels) - 2:
        mark[verb_index + 2] = 1
        ctx_p2 = sentence[verb_index + 2]
    else:
        ctx_p2 = 'eos'
    
    word_idx = [srl_word_dict.get(w, UNK_IDX) for w in sentence]

    ctx_n2_idx = [srl_word_dict.get(ctx_n2, UNK_IDX)] * sen_len
    ctx_n1_idx = [srl_word_dict.get(ctx_n1, UNK_IDX)] * sen_len
    ctx_0_idx = [srl_word_dict.get(ctx_0, UNK_IDX)] * sen_len
    ctx_p1_idx = [srl_word_dict.get(ctx_p1, UNK_IDX)] * sen_len
    ctx_p2_idx = [srl_word_dict.get(ctx_p2, UNK_IDX)] * sen_len

    if predicate in srl_verb_dict:
        pred_idx = [srl_verb_dict.get(predicate)] * sen_len
    else:
        print "predicate %s not in dictionary. using UNK_IDX  " % predicate
        pred_idx = [UNK_IDX] * sen_len


    label_idx = ['0' for w in sentence]    
    data= word_idx, ctx_n2_idx, ctx_n1_idx, \
          ctx_0_idx, ctx_p1_idx, ctx_p2_idx, pred_idx, mark, label_idx
    
    return data

def testPI(pi_inferer,srl_inferer):
    sentence="昨天 我 送给 朋友 一千 元 的 礼物"
    tokens=sentence.split()
    word_idx=[pi_word_dict.get(pi_data_feeder.canonicalize_word(w,pi_word_dict),UNK_IDX) for w in tokens]
    #word_idx=[0 for w in tokens]
    mark=[0 for w in tokens]
    label_idx=['O' for w in tokens]

    l=word_idx,mark

    pi_labels_reverse={}
    for(k,v) in pi_label_dict.items():
        pi_labels_reverse[v]=k
    print pi_labels_reverse    

    print l
    lab_ids=pi_inferer.infer(input=[l],field='id')
    pre_lab=[pi_labels_reverse[lab_id] for lab_id in lab_ids]
    print pre_lab

    counter=0
    predicateList=[]
    for lab in pre_lab:
        if lab=='B-V':
            labels=['O' for lab in pre_lab]
            labels[counter]='B-V'
            predicateList.append(labels)
        counter=counter+1    
    print predicateList

    

    
    # get string labels
    srl_labels_reverse = {}
    for (k, v) in srl_label_dict.items():
        srl_labels_reverse[v] = k

     
    

    for predicates in predicateList:
        data=featureGenHelper(tokens,predicates)
        probs = srl_inferer.infer(input=[data], field='id')
        pre_lab = [srl_labels_reverse[i] for i in probs]
        print pre_lab




def main():
 paddle.init(use_gpu=False, trainer_count=1)
 # load pi model
 pi_inferer=loadPredicateIdentifierModel()
# load SRL model
 srl_inferer=loadSRLModel()
 testPI(pi_inferer,srl_inferer) 

if __name__ == '__main__':
    main() 

