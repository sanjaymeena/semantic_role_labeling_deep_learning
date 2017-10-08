import math
import numpy as np
import gzip
import paddle.v2 as paddle
from  paddle.v2.parameters import  Parameters 
from paddle.trainer_config_helpers import *
import itertools
from collections import OrderedDict
import time


#import paddle.v2.evaluator as evaluator
import srl_data_feeder as srl_data_test
import tarfile
import logging

word_dict, verb_dict, label_dict = srl_data_test.get_dict()
word_dict_len = len(word_dict)
label_dict_len = len(label_dict)
pred_len = len(verb_dict)

mark_dict_len = 2
word_dim = 50
mark_dim = 5
hidden_dim = 512
depth = 8
default_std = 1 / math.sqrt(hidden_dim) / 3.0
mix_hidden_lr = 1e-3

UNK_IDX = 0

#model_file="models/srl_params_pass_530_100517.tar.gz"
#model_file="models/srl_params_pass_420_090617.tar.gz"
model_file="models/parameters/srl/srl_params_pass_150.tar.gz"



def load_parameter(file_name, h, w):
    with open(file_name, 'rb') as f:
        f.read(16)  # skip header.
        return np.fromfile(f, dtype=np.float32).reshape(h, w)

def d_type(size):
    return paddle.data_type.integer_value_sequence(size)


def db_lstm():
    #8 features
    word = paddle.layer.data(name='word_data', type=d_type(word_dict_len))
    # we are changing the pred_len to word_dict_len for testing purpose
    predicate = paddle.layer.data(name='verb_data', type=d_type(word_dict_len))
    #predicate = paddle.layer.data(name='verb_data', type=d_type(pred_len))


    ctx_n2 = paddle.layer.data(name='ctx_n2_data', type=d_type(word_dict_len))
    ctx_n1 = paddle.layer.data(name='ctx_n1_data', type=d_type(word_dict_len))
    ctx_0 = paddle.layer.data(name='ctx_0_data', type=d_type(word_dict_len))
    ctx_p1 = paddle.layer.data(name='ctx_p1_data', type=d_type(word_dict_len))
    ctx_p2 = paddle.layer.data(name='ctx_p2_data', type=d_type(word_dict_len))
    mark = paddle.layer.data(name='mark_data', type=d_type(mark_dict_len))

    emb_para = paddle.attr.Param(name='emb', initial_std=0., is_static=True)
    std_0 = paddle.attr.Param(initial_std=0.)
    std_default = paddle.attr.Param(initial_std=default_std)

    predicate_embedding = paddle.layer.embedding(
        size=word_dim,
        input=predicate,
        param_attr=paddle.attr.Param(name='vemb', initial_std=default_std))
    mark_embedding = paddle.layer.embedding(
        size=mark_dim, input=mark, param_attr=std_0)

    word_input = [word, ctx_n2, ctx_n1, ctx_0, ctx_p1, ctx_p2]
    emb_layers = [
        paddle.layer.embedding(size=word_dim, input=x, param_attr=emb_para)
        for x in word_input
    ]
    emb_layers.append(predicate_embedding)
    emb_layers.append(mark_embedding)

    hidden_0 = paddle.layer.mixed(
        size=hidden_dim,
        bias_attr=std_default,
        input=[
            paddle.layer.full_matrix_projection(
                input=emb, param_attr=std_default) for emb in emb_layers
        ])

    lstm_para_attr = paddle.attr.Param(initial_std=0.0, learning_rate=1.0)
    hidden_para_attr = paddle.attr.Param(
        initial_std=default_std, learning_rate=mix_hidden_lr)

    lstm_0 = paddle.layer.lstmemory(
        input=hidden_0,
        act=paddle.activation.Relu(),
        gate_act=paddle.activation.Sigmoid(),
        state_act=paddle.activation.Sigmoid(),
        bias_attr=std_0,
        param_attr=lstm_para_attr)

    #stack L-LSTM and R-LSTM with direct edges
    input_tmp = [hidden_0, lstm_0]

    for i in range(1, depth):
        mix_hidden = paddle.layer.mixed(
            size=hidden_dim,
            bias_attr=std_default,
            input=[
                paddle.layer.full_matrix_projection(
                    input=input_tmp[0], param_attr=hidden_para_attr),
                paddle.layer.full_matrix_projection(
                    input=input_tmp[1], param_attr=lstm_para_attr)
            ])

        lstm = paddle.layer.lstmemory(
            input=mix_hidden,
            act=paddle.activation.Relu(),
            gate_act=paddle.activation.Sigmoid(),
            state_act=paddle.activation.Sigmoid(),
            reverse=((i % 2) == 1),
            bias_attr=std_0,
            param_attr=lstm_para_attr)

        input_tmp = [mix_hidden, lstm]

    feature_out = paddle.layer.mixed(
        size=label_dict_len,
        bias_attr=std_default,
        input=[
            paddle.layer.full_matrix_projection(
                input=input_tmp[0], param_attr=hidden_para_attr),
            paddle.layer.full_matrix_projection(
                input=input_tmp[1], param_attr=lstm_para_attr)
        ], )

    return feature_out


def featureGenHelper(sentence,predicate,labels):
    sen_len = len(sentence)
    #print sentence,predicate,new_labels
    labels
    if 'B-V' not in labels:
        print 'B-V not present : ', predicate,labels,sentence
    verb_index = labels.index('B-V')
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
    word_idx = [word_dict.get(w, UNK_IDX) for w in sentence]

    ctx_n2_idx = [word_dict.get(ctx_n2, UNK_IDX)] * sen_len
    ctx_n1_idx = [word_dict.get(ctx_n1, UNK_IDX)] * sen_len
    ctx_0_idx = [word_dict.get(ctx_0, UNK_IDX)] * sen_len
    ctx_p1_idx = [word_dict.get(ctx_p1, UNK_IDX)] * sen_len
    ctx_p2_idx = [word_dict.get(ctx_p2, UNK_IDX)] * sen_len

    if predicate in verb_dict:
        pred_idx = [verb_dict.get(predicate)] * sen_len
    else:
        print "predicate %s not in dictionary. using UNK_IDX  " % predicate
        pred_idx = [verb_dict.get(predicate,UNK_IDX)] * sen_len


    label_idx = [label_dict.get(w) for w in labels]    
    data= word_idx, ctx_n2_idx, ctx_n1_idx, \
          ctx_0_idx, ctx_p1_idx, ctx_p2_idx, pred_idx, mark, label_idx
    
    return word_idx, ctx_n2_idx, ctx_n1_idx, \
          ctx_0_idx, ctx_p1_idx, ctx_p2_idx, pred_idx, mark, label_idx
        
    
def readTokenizedSentence(sentence):
    sentence= sentence.replace('\n','')
    #print sentence
    sents=[]
    tokens=sentence.split()
    labels=['O']*len(tokens)

    data=[]
    counter=0
    for tok in tokens:
        if tok in verb_dict:
            datum=[]
            #print 'predicate found in dict',tok,counter
            new_labels=list(labels)
            predicate=tok
            new_labels[counter]='B-V'
            #print sentence, predicate,new_labels
            datum.append(tokens)
            datum.append(predicate)
            datum.append(new_labels)
            
            data.append(datum)
        counter+=1    
    return data        
    
def genSentencePerPredicate(data_list):
    mulSentence=[]
    for datalist in data_list:
        tokens=datalist[0]
        predicate=datalist[1]
        labels=datalist[2]
        #print tokens,predicate,labels
        feature_sen=featureGenHelper(tokens,predicate,labels)
        # add this feature_sen again to a list
        l=[]
        l.append(feature_sen)
        #print l
        #print ''
        mulSentence.append(l)
    return mulSentence    

def genModelRelatedData(sentence):
    mulSentence=[]
    data_list=readTokenizedSentence(sentence)
    mulSentence=genSentencePerPredicate(data_list)
    return mulSentence
    

# This function will return a dictioary . Key =sentence and value is sentence features per predicate
def readTokenizedSentences(data_file):
    sentences = []
    labels = []
    one_seg = []
    counter=0
    all_data=[]
    my_dict =OrderedDict()
    with open(data_file) as data: 
        for sentence in data: 
            sentences=genModelRelatedData(sentence)
            #print sentences
            my_dict[sentence]=sentences
            
    data.close()
    return my_dict


# This function can read the data features of sentences in format of 
# word_idx, ctx_n2_idx, ctx_n1_idx, \
# ctx_0_idx, ctx_p1_idx, ctx_p2_idx, pred_idx, mark, label_idx
def readDatawithStringFeatures(dataFile):
    data_dict=OrderedDict()

    data_set=OrderedDict()
    orig_counter=0
    with open(dataFile) as data: 
        tokenizedSentences=[]
        datarow=[]
        for sentence in data:
            orig_counter+=1
            if len(sentence.strip()) > 1 :
                data_set[sentence]=sentence
                

    data.close()

    tokenizedSentences=[]
    set_counter=0
    for sentence in data_set.keys():
        set_counter+=1
        if len(sentence.strip()) > 1 :
            sentence=sentence.replace('\n','')
            #print sentence
            vals=sentence.split('\t')
            tokenizedSentence=vals[0]
            if tokenizedSentence not in  data_dict:
                newList=[]
                newList.append(sentence)
                data_dict[tokenizedSentence]=newList
            else:
                data_dict.get(tokenizedSentence).append(sentence)


    #print 'original data size ', orig_counter, 'final data size', set_counter           
    # for key in data_dict:
    #     print 'sentence: ', key
    #     for value in data_dict.get(key):
    #         print '    ',value, 'total val elements : ', len(vals)


    return data_dict

def getFeatureData(sentence,ctx_n2,ctx_n1,ctx_0,ctx_p1,ctx_p2,predicate,mark,labels):
    sen_len=len(sentence)
    word_idx = [word_dict.get(w, UNK_IDX) for w in sentence]

    ctx_n2_idx = [word_dict.get(ctx_n2, UNK_IDX)] * sen_len
    ctx_n1_idx = [word_dict.get(ctx_n1, UNK_IDX)] * sen_len
    ctx_0_idx = [word_dict.get(ctx_0, UNK_IDX)] * sen_len
    ctx_p1_idx = [word_dict.get(ctx_p1, UNK_IDX)] * sen_len
    ctx_p2_idx = [word_dict.get(ctx_p2, UNK_IDX)] * sen_len

    if predicate in verb_dict:
        pred_idx = [verb_dict.get(predicate)] * sen_len
    else:
        print "predicate %s not in dictionary. using UNK_IDX  " % predicate
        pred_idx = [verb_dict.get(predicate,UNK_IDX)] * sen_len


    label_idx = [label_dict.get(w) for w in labels]
    
    return word_idx, ctx_n2_idx, ctx_n1_idx, \
         ctx_0_idx, ctx_p1_idx, ctx_p2_idx, pred_idx, mark, label_idx

# This function generates the final features that are used as input to the training feature. 
def genNumericModelFeatures(my_dict):
    new_dict=OrderedDict()
    for key in my_dict:
        #print key

        featureList=[]
        for value in my_dict.get(key):
            
            vals=value.split('\t')
            #print len(vals)
            #print value, 'total val elements : ', len(vals)
            word_idx= vals[0].split()
            ctx_n2_idx= vals[1]
            ctx_n1_idx=vals[2]
            ctx_0_idx= vals[3]
            ctx_p1_idx= vals[4]
            ctx_p2_idx=vals[5]
            pred_idx= vals[6]
            #print vals[7].split()
            mark=map(int,vals[7].split())
            label_idx=vals[8].split()


            word_idx, ctx_n2_idx, ctx_n1_idx, \
             ctx_0_idx, ctx_p1_idx, ctx_p2_idx, pred_idx, mark, label_idx\
                =getFeatureData(word_idx,ctx_n2_idx,ctx_n1_idx,ctx_0_idx,ctx_p1_idx,ctx_p2_idx,pred_idx,mark,label_idx)

            feature_sen= word_idx, ctx_n2_idx, ctx_n1_idx, \
             ctx_0_idx, ctx_p1_idx, ctx_p2_idx, pred_idx, mark, label_idx

            #print word_idx, ctx_n2_idx, ctx_n1_idx, \
            #  ctx_0_idx, ctx_p1_idx, ctx_p2_idx, pred_idx, mark, label_idx
            l=[]
            l.append(feature_sen)
            featureList.append(l)

        new_dict[key]=featureList
    return new_dict

# converts labels from BIO format to conll format    
def convertBIOtoCONLL(labelList):

    output=[]
    counter=0


    lastB=''
    for token in labelList:
        prevToken=''
        label=''
        temp=''

        if counter > 1:
            prevToken=labelList[counter-1]


        if token == 'O':
            label= '*'
        elif token=='B-V':
            temp=token.replace('B-','')
            label ='(' + temp + '*' + ')'
        elif 'B-' in token:
            
            temp=token.replace('B-','')
            lastB=temp
            
            if counter < len(labelList)-1:
                next_tok=labelList[counter+1]
                if ('B-' in next_tok) or ('O' == next_tok):
                    label='(' + temp + '*' + ')' 
                else:
                    label='(' + temp + '*'
            elif counter==len(labelList)-1:
                    label='(' + temp + '*' + ')'      
        elif 'I-' in token:
            temp1=token.replace('I-','')
            if counter < len(labelList)-1:
                next_tok=labelList[counter+1]
                if 'I-' in next_tok:
                    label='*'
                elif (lastB==temp1) and (lastB !='') and (prevToken!='O'):
                    label='*)'
                else:
                    label='*'
                    #print 'please check this output ' ,labelList  
                    dummy=10  
            elif counter==len(labelList)-1:
                label='*)'
        counter+=1           
                    
        output.append(label)
        if label=='':
            print 'label is empty for : ', token , ' tokenList: ',labelList
    #print output, 'total conll labels : ' , len(output), ' total labels : ', len(labelList)  , labelList 
    return output  

# load model
# model parameters

def loadModel():
    logger = logging.getLogger('paddle')
    logger.setLevel(logging.WARN)
    paddle.init(use_gpu=False, trainer_count=1)

    feature_out = db_lstm()


    #f="srl_params_pass_530_100517.tar.gz"
    
    #load parameters
    print 'extracting  srl db-lstm model parameters from : ', model_file ,' ...'
    with gzip.open(model_file, 'r') as f:
        parameters = Parameters.from_tar(f)


    predict = paddle.layer.crf_decoding(
        size=label_dict_len,
        input=feature_out,
        param_attr=paddle.attr.Param(name='crfw'))


    model_parameters=parameters
    model_predict=predict


    return model_parameters,model_predict


#prediction_layer=loadModel();
parameters,predict=loadModel();
print 'initializing srl db-lstm model ..'
inferer = paddle.inference.Inference(
                 output_layer=predict, parameters=parameters)
print 'done loading the srl db-lstm  model.'

# do prediction on input data
def srl_predict(inputData):



    # logger = logging.getLogger('paddle')
    # logger.setLevel(logging.WARN)
    # paddle.init(use_gpu=False, trainer_count=1)

    # feature_out = db_lstm()

    

    # f="srl_params_pass_420_090617.tar.gz"
    
    # #load parameters

    # with gzip.open(f, 'r') as f:
    #     parameters = Parameters.from_tar(f)

    # predict = paddle.layer.crf_decoding(
    #     size=label_dict_len,
    #     input=feature_out,
    #     param_attr=paddle.attr.Param(name='crfw'))

    # do prediction
   
    # we will try new version faster version of inferer
   
    # probs = paddle.infer(
    #             output_layer=predict,
    #             parameters=parameters,
    #             input=inputData,
    #             field='id')

    
    # inferer = paddle.inference.Inference(
    #              output_layer=predict, parameters=parameters)

    probs = inferer.infer(input=inputData, field='id')

    
    # get string labels
    labels_reverse = {}
    for (k, v) in label_dict.items():
        labels_reverse[v] = k

     
    pre_lab = [labels_reverse[i] for i in probs]
        
    
    return pre_lab

"""
This function is for validation of sequence of labels :

* I-Tag should be preceeded by B-Tag somewhere in the sequence
* B-V should be present
"""
def validateLabelSequence(labelList):
    newlabelList=[]
    isValid=True
    hasVerb=False;
    msg=''
    counter=0
    lastB=''
    for token in labelList:
        label=''
        temp=''
        prevToken=''

        if counter > 1:
            prevToken=labelList[counter-1]
        
        if token == 'O':
            newToken=token
        elif token=='B-V':
            hasVerb=True
            newToken=token
        elif 'B-' in token:
            lastB=token.replace('B-','')
            newToken=token
            #print 'lastB :', lastB
        elif 'I-' in token:
            #print token
            iTag=token.replace('I-','')
            #print temp
            if lastB != iTag or prevToken== 'B-V':
                msg= 'error: ' + token +' without B-'+iTag+ ' tag'
                isValid=False
                newToken='O'
            else:
                newToken=token
        counter+=1 
        newlabelList.append(newToken)  
        
    if not hasVerb:
        msg+=' error: no B-V tag'
        isValid=False
    if not isValid:
             print 'the sequence : ', labelList , ' is not valid ', msg   
    #print   newlabelList 
    return newlabelList




# generate conll format data
def generateCONLLFormat(words,argumentColumns):
    conll=[]
    count=0
    conll09=''
    
    
    for item in words:
        #print item
        row=[]
        row.append(item)
        #print count
        
        for a in argumentColumns:
            row.append(a[count])
            
        
        conll.append(row)
        count+=1
        
    for row in conll:
        count=0
        for item in row:
            conll09+=str(item) 
            if count < len(row):
                count+=1
                conll09+='\t'
        conll09+='\n'
    return conll09   

# generate prop data
def generatePropData(words,argumentColumns):
    conll=[]
    conll05=''
    
    string=''
    for w in words:
        string+= w +  ' '
    #print len(words), '  ', string    
    predRow=[srl_data_test.conll_empty for i in range(len(words))]
    #print 'init predrow ' , len(predRow), ' '  , predRow   

    #generate the predicate row
    #print 'arguments info :: ', len(argumentColumns), ' ', argumentColumns
    for arr in argumentColumns:
        count=0
        for a in arr:

            if '(V*)'== a :
                predRow[count]= words[count]
            
            count+=1

    #print len(predRow), " " ,predRow
    counter=0 
    for item in predRow:
        #print item
        row=[]
        row.append(item)
        #print count
        #print counter, 'len of argumentColumns : ', len(argumentColumns)
        for a in argumentColumns:
            #print 'length of argument: '  , len(a)
            row.append(a[counter])
            
        conll.append(row)
        counter+=1
            

        
    for row in conll:
        count=0
        rowData=''
        for item in row:
            rowData+=str(item) 
            if count < len(row)-1:
                count+=1

                rowData+='\t'
        rowData=rowData.strip()
        conll05+=rowData        
        conll05+='\n'
    #print conll05    
    return conll05   
    
    



def testCode():
    logger = logging.getLogger('paddle')
    logger.setLevel(logging.WARN)
    paddle.init(use_gpu=False, trainer_count=1)

    feature_out = db_lstm()
    # target = paddle.layer.data(name='target', type=d_type(label_dict_len))
    # crf_cost = paddle.layer.crf(
    #     size=label_dict_len,
    #     input=feature_out,
    #     label=target,
    #     param_attr=paddle.attr.Param(
    #         name='crfw', initial_std=default_std, learning_rate=mix_hidden_lr))

    # crf_dec = paddle.layer.crf_decoding(
    #     size=label_dict_len,
    #     input=feature_out,
    #     label=target,
    #     param_attr=paddle.attr.Param(name='crfw'))
    # evaluator.sum(input=crf_dec)

    # # create parameters
    # parameters1 = paddle.parameters.create(crf_cost)
    # parameters1.set('emb', load_parameter(srl_data.get_embedding(), 218625, 150))
    # model file
    # f="parameters/params_pass_15.tar.gz"
    
    
    
    #load parameters
    #parameters1=parameters1.from_tar(f)
    with gzip.open(model_file, 'r') as f:
        parameters = Parameters.from_tar(f)
    

    

  
    predict = paddle.layer.crf_decoding(
        size=label_dict_len,
        input=feature_out,
        param_attr=paddle.attr.Param(name='crfw'))

    # test_creator = srl_data.test()
    # test_data = []
    # for item in test_creator():
    #     test_data.append(item)
    #     if len(test_data) == 1:
    #         break

    # probs = paddle.infer(
    #     output_layer=predict,
    #     parameters=parameters,
    #     input=test_data,
    #     field='id')

    # print test_data
    # for x in test_data:
    #     for y in x:
    #         print y
    # print 'label', test_data[0][8]
    # res=chunk_evaluator(probs,test_data[0][8],"IOB",2)
    # print res

    # assert len(probs) == len(test_data[0][0])
    # labels_reverse = {}
    # for (k, v) in label_dict.items():
    #     labels_reverse[v] = k
    # pre_lab = [labels_reverse[i] for i in probs]
    # print pre_lab


    print 'lets run inference on all the data examples'
    test_datax=[]
    test_creator = srl_data_test.test()
    gold_answers=[]
    for item in test_creator():
       test_data = []
       test_data.append(item[0:8])
       gold_answers.append(item[8])
       test_datax.append(test_data)

    # for x in gold_answers:
    #     print x   
    answers=[]
    test_datax=test_datax[0:2]

    
          
    for element in test_datax:
        #print element
        probs = paddle.infer(
                output_layer=predict,
                parameters=parameters,
                input=element,
                field='id')
        print probs
        #assert len(probs) == len(element[0][0])
        labels_reverse = {}
        for (k, v) in label_dict.items():
            labels_reverse[v] = k
        pre_lab = [labels_reverse[i] for i in probs]
        answers.append(pre_lab)

    counter=0
    total_predicts=0
    predicts=""
    for orig,pred in itertools.izip(gold_answers,answers):
        
        labels_reverse = {}
        for (k, v) in label_dict.items():
            labels_reverse[v] = k
        pre_lab = [labels_reverse[i] for i in orig]

        #print pre_lab,pred
        predicts+=str(pre_lab).strip('[]') + "\t" + str(pred).strip('[]') + "\n"

        
        for x,y in itertools.izip(pre_lab,pred):
            total_predicts=total_predicts+1
            #print x,y
            if x == y:
                counter=counter+1
    acc=counter/(total_predicts*1.0)
    print counter,total_predicts
    print 'accuracy is :' + str(acc *100 )+ '%'

    result=""
    result+='correct labels : ' + str(counter) + ',' + ' , incorrect labels :' + str(total_predicts) +"\n"
    result+='accuracy is : ' + str(acc * 100) +" % \n"
    predicts+= result 

    target = open("out/test_results.txt", 'w')
    target.write(predicts)
    target.close() 



def testCode1():
    data_file='data/tokenized_sentences.txt'   
    all_data=readTokenizedSentences(data_file)
    

    for key in all_data.keys():
        sentence=all_data.get(key)
        argumentColumns=[]

        words=key.split()
 
  # below is example given that arguments are obtained
        for senFeatures in sentence:

            #print senFeatures
            prediction=srl_predict(senFeatures)
            argumentColumns.append(prediction)
            
        conll09=generateCONLLFormat(words,argumentColumns)
        print conll09




def getTextLabels(index_labels):
    
    textLabels=[]

    labels_reverse = {}
    for (k, v) in label_dict.items():
        labels_reverse[v] = k
    #print index_labels    
    
    for labels in index_labels:
        #print predicted
         text_lab = [labels_reverse[i] for i in labels]
         textLabels.append(text_lab)
    return textLabels     



# generate Statistics 
def generateStats(gold_answers,predicted_answers):
    counter=0
    total_predicts=0
    predicts=""

    total_sentence_counter=0
    for gold,predicted in itertools.izip(gold_answers,predicted_answers):
        total_sentence_counter+=1
        for g,p in itertools.izip(gold,predicted):
            
            total_predicts=total_predicts+1
            #print x,y
            if g == p:
                counter=counter+1
    acc=counter/(total_predicts*1.0)
    print counter,total_predicts
    print 'accuracy is :' + str(acc *100 )+ '%'

    result=""

    result+='correct labels : ' + str(counter) + ',' + ' , total labels :' + str(total_predicts) +"\n"
    result+='accuracy is : ' + str(acc * 100) +" % \n"
    

    return result    

def testCode2():
    #intermediateFile='data/test/test_all.txt'    
    intermediateFile='data/test/test.txt'    
    #intermediateFile='data/train/train_sample.txt'    
    my_dict=readDatawithStringFeatures(intermediateFile)
    new_dict=genNumericModelFeatures(my_dict)    
    
    
    predictedData=""
    propData=""
    gold_answers=[]
    predicted_answers=[]

    conllList=[]
    propList=[]

    conlloutputlist=[]
    predictedLabelList=[]

    total_sentences_counter=0
    for key in new_dict.keys():
        total_sentences_counter+=1
        sentence=new_dict.get(key)
        argumentColumns=[]

        words=key.split()
 
  # below is example given that arguments are obtained
        for senFeatures in sentence:

            
            #print senFeatures[0][8]
            gold_ans=senFeatures[0][8]


            #print senFeatures
            prediction=srl_predict(senFeatures[0:7])


            # add gold answers to list
            gold_answers.append(gold_ans)
            # add predicted answers to list


           
            # now we add to predicted answers list
            predicted_answers.append(prediction)

            predictedLabelList.append(prediction)
        
             # validate the label sequence. this function can remove error sequence with correct 
            prediction=validateLabelSequence(prediction)

            #convert bio format to conll format
            prediction=convertBIOtoCONLL(prediction)
            conlloutputlist.append(prediction)
            # append to argument column
            argumentColumns.append(prediction)
            
        conll09=generateCONLLFormat(words,argumentColumns)
        
        conll05=generatePropData(words,argumentColumns)
        # append to conll list
        conllList.append(conll09)

        # append to conll05 list
        propList.append(conll05)
        
    for conll in conllList:
        predictedData+=conll + "\n"


    for conll05 in propList:
        propData+=conll05 + "\n"

    gold_answers=getTextLabels(gold_answers)    

    stats=generateStats(gold_answers,predicted_answers)


    total_sen='total number of sentences : ' + str(total_sentences_counter)

    print predictedData
    print stats
    print total_sen

    
    finalOutput=predictedData +'\n' + stats + "\n" + total_sen
    target = open("out/test_results.txt", 'w')
    target.write(finalOutput)
    target.close() 

    target = open("out/test_prop_results.txt", 'w')
    target.write(propData)
    target.close() 

    builder=''
    
    for predictedLabels,output in itertools.izip(predictedLabelList , conlloutputlist) :
        string1=''
        string2=''

        
        for tag in predictedLabels:
            string1+= tag + ' '

        for tag in output:
            string2+= tag + '\t'    
                
        builder+=string1 + '\t' + string2
        builder+='\n'
    print builder
    target = open("out/validate_tags.txt", 'w')
    target.write(builder)
    target.close() 

            
            
            
def evaluationDataGenerator2(eval_data_type='test'):
    print 'evaluation on data :' + eval_data_type
    if eval_data_type=='test':
        test_creator = srl_data_test.test()
    else:
        test_creator = srl_data_test.train()    

    test_data = []
    gold_answers=[]
    for item in test_creator():
       
       test_data.append(item[0:8])
       gold_answers.append(item[8])
    
    builder=""



  # below is example given that arguments are obtained
    sencounter = 0
    for senFeatures in test_data:

        words=senFeatures[0]
        #print senFeatures[0][8]
        gold_ans=gold_answers[sencounter]
        


        #print senFeatures
        prediction=srl_predict([senFeatures])


        
        gold_ans_text_labels=getTextLabels([gold_ans])[0] 
        #print gold_ans_text_labels

        temp=[]
        temp.append(prediction)
        

        counter=0
        for w in words:
            gold=gold_ans_text_labels[counter]
            pred=prediction[counter]

            string=str(w) + " " + gold + " " +pred+"\n"
            builder+=string
            counter=counter+1
        builder+="\n" 
        sencounter=sencounter+1   


    #print builder
    target = open("out/eval_test.txt", 'w')
    target.write(builder)
    target.close()  
    print 'total data in eval set: ' + str(len(test_data))

def evaluationDataGenerator():


    #intermediateFile='data/test/test_all.txt'    
    intermediateFile='data/srl/test/test.txt'    
    #intermediateFile='data/train/train_sample.txt'    
    my_dict=readDatawithStringFeatures(intermediateFile)
    new_dict=genNumericModelFeatures(my_dict)    
    
    builder=""

    total_sentences_counter=0
    for key in new_dict.keys():
        total_sentences_counter+=1
        sentence=new_dict.get(key)
        argumentColumns=[]

        words=key.split()

  # below is example given that arguments are obtained
        for senFeatures in sentence:

            
            #print senFeatures[0][8]
            gold_ans=senFeatures[0][8]


            #print senFeatures
            prediction=srl_predict(senFeatures[0:7])


            temp=[]
            temp.append(gold_ans)
            gold_ans_text_labels=getTextLabels(temp)[0] 

            temp=[]
            temp.append(prediction)
            
        
            counter=0
            for w in words:
                gold=gold_ans_text_labels[counter]
                pred=prediction[counter]

                string=w + " " + gold + " " +pred+"\n"
                builder+=string
                counter=counter+1
            builder+="\n"    

  
    #print builder
    target = open("out/eval_test.txt", 'w')
    target.write(builder)
    target.close() 

def main():
    # test_creator = srl_data_test.test()
    # test_data = []
    # for item in test_creator():
    #     test_data.append(item)
    #     if len(test_data) == 1:
    #         break

    # print 'now doing prediction'
    # print test_data
    # prediction=srl_predict(test_data)    
    # print prediction  
    start_time = time.time()
    #testCode2()

    eval_data_type='test'

    evaluationDataGenerator2(eval_data_type)
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    main()


