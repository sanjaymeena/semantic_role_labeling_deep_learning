import tarfile
import gzip
import itertools
import random 
import numpy as np
"""
SRL Dataset reader
"""



UNK_IDX = 0

word_dict_file = './data/embedding/vocab.txt'
embedding_file='./data/embedding/wordVectors.txt'

word_dict_srl_file = './data/srl/train/words_SRL.txt'
prop_SRL_file='./data/srl/train/prop_SRL.txt'
predicate_file = './data/srl/train/verbDict_SRL.txt'

test_word_dict_srl_file = './data/srl/test/words_SRL.txt'
test_prop_SRL_file='./data/srl/test/prop_SRL.txt'
test_predicate_file = predicate_file


label_dict_file = './data/srl/train/targetDict.txt'


train_list_file = './data/srl/train.list'
test_list_file = './data/srl/test.list'


# there is a line to line mapping between embedding file and word_dict_file



conll_empty='_'

def load_dict(filename):
    d = dict()
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            d[line.strip()] = i
    return d


# this is a test function. not done yet
# def generateconllCorpus(words_file, props_file):
    

#         sentences = []
#         labels = []
#         one_seg = []
#         counter=0
#         with open(words_file) as w, open(props_file) as p: 
#             for word, label in itertools.izip(w, p):
#                 word = word.strip()
#                 label = label.strip().split()
                

#                 if len(label) == 0:  # end of sentence
#                     dummy=0
                    

#         w.close()
#         p.close()              
      
                
#     return reader




def corpus_reader(words_file, props_file):
    """
    Read one corpus. It returns an iterator. Each element of
    this iterator is a tuple including sentence and labels. The sentence is
    consist of a list of word IDs. The labels include a list of label IDs.
    :return: a iterator of data.
    :rtype: iterator
    """

    def reader():
       
        sentences = []
        labels = []
        one_seg = []
        counter=0
        with open(words_file) as w, open(props_file) as p: 
            for word, label in itertools.izip(w, p):
                word = word.strip()
                label = label.strip().split()
                

                if len(label) == 0:  # end of sentence
                    for i in xrange(len(one_seg[0])):
                        a_kind_lable = [x[i] for x in one_seg]
                        labels.append(a_kind_lable)

                    if len(labels) >= 1:
                        verb_list = []
                        for x in labels[0]:
                            if x != conll_empty:
                                verb_list.append(x)

                        for i, lbl in enumerate(labels[1:]):
                            cur_tag = 'O'
                            is_in_bracket = False
                            lbl_seq = []
                            verb_word = ''
                            for l in lbl:
                                if l == '*' and is_in_bracket == False:
                                    lbl_seq.append('O')
                                elif l == '*' and is_in_bracket == True:
                                    lbl_seq.append('I-' + cur_tag)
                                elif l == '*)':
                                    lbl_seq.append('I-' + cur_tag)
                                    is_in_bracket = False
                                elif l.find('(') != -1 and l.find(')') != -1:
                                    cur_tag = l[1:l.find('*')]
                                    lbl_seq.append('B-' + cur_tag)
                                    is_in_bracket = False
                                elif l.find('(') != -1 and l.find(')') == -1:
                                    cur_tag = l[1:l.find('*')]
                                    lbl_seq.append('B-' + cur_tag)
                                    is_in_bracket = True
                                else:
                                    raise RuntimeError('Unexpected label: %s' %
                                                       l)
                            #print sentences, verb_list[i], lbl_seq
                            
                            yield sentences, verb_list[i], lbl_seq

                    sentences = []
                    labels = []
                    one_seg = []
                    counter+=1
                    
                else:
                    sentences.append(word)
                    one_seg.append(label)

        w.close()
        p.close()              
      
                
    return reader


def reader_creator(corpus_reader,
                   word_dict=None,
                   predicate_dict=None,
                   label_dict=None):
    def reader():
        counter=0
        for sentence, predicate, labels in corpus_reader():
            counter+=1
            sen_len = len(sentence)

            if 'B-V' not in labels:
                print 'B-V not present : ', predicate,labels

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

			# we are changing from predicate_dict to word_dict (embedding) for testing
            if predicate in word_dict:
                pred_idx = [predicate_dict.get(predicate)] * sen_len
            else:
                print "predicate %s not in dictionary. using UNK_IDX  " % predicate
                pred_idx = [predicate_dict.get(predicate,UNK_IDX)] * sen_len




            label_idx = [label_dict.get(w) for w in labels]

            # print 'sentence id: ', counter
            # for string in sentence:
            #     print string
            # print '/n'
            # print predicate,labels
            # print ''
            # #print counter, word_idx, label_idx
            # print word_idx, ctx_n2_idx, ctx_n1_idx, \
            #   ctx_0_idx, ctx_p1_idx, ctx_p2_idx, pred_idx, mark, label_idx

            yield word_idx, ctx_n2_idx, ctx_n1_idx, \
              ctx_0_idx, ctx_p1_idx, ctx_p2_idx, pred_idx, mark, label_idx

    return reader


def get_dict():
    word_dict = load_dict(word_dict_file)
    verb_dict = load_dict(predicate_file)
    label_dict = load_dict(label_dict_file)
    return word_dict, verb_dict, label_dict

def get_test_dict():
    word_dict = load_dict(word_dict_file)
    verb_dict = load_dict(predicate_file)
    label_dict = load_dict(label_dict_file)
    return word_dict, verb_dict, label_dict    

   
def get_embedding(emb_file=embedding_file):
    """
    Get the trained word vector.
    """
    return np.loadtxt(emb_file, dtype=float)
    
def shuffle(reader, buf_size):
    """
    Creates a data reader whose data output is shuffled.

    Output from the iterator that created by original reader will be
    buffered into shuffle buffer, and then shuffled. The size of shuffle buffer
    is determined by argument buf_size.

    :param reader: the original reader whose output will be shuffled.
    :type reader: callable
    :param buf_size: shuffle buffer size.
    :type buf_size: int

    :return: the new reader whose output is shuffled.
    :rtype: callable
    """

    def data_reader():
        buf = []
        for e in reader():
            buf.append(e)
            if len(buf) >= buf_size:
                random.shuffle(buf)
                for b in buf:
                    yield b
                buf = []

        if len(buf) > 0:
            random.shuffle(buf)
            for b in buf:
                yield b

    return data_reader    


def train():
    word_dict, verb_dict, label_dict = get_dict()
   
    reader = corpus_reader(
        words_file=word_dict_srl_file,
        props_file=prop_SRL_file)
    return reader_creator(reader, word_dict, verb_dict, label_dict)


def test():
    word_dict, verb_dict, label_dict = get_test_dict()
   
    reader = corpus_reader(
        words_file=test_word_dict_srl_file,
        props_file=test_prop_SRL_file)
    return reader_creator(reader, word_dict, verb_dict, label_dict)
    
def main():
  
  #reader = corpus_reader(word_dict_srl_file,prop_SRL_file)
  #word_dict, verb_dict, label_dict = get_dict()

  reader1 = corpus_reader(
        words_file=word_dict_srl_file,
        props_file=prop_SRL_file)
  c=1
  for x in reader1():
    c+=1
  print c 
  target = open("train/train_data.txt", 'w')
  reader=train();
  counter=0
  for x in reader():
    target.write(str(x))
    target.write(str("\n"))
    counter+=1
  target.close() 
  print 'total train sentences : ', counter

  reader2 = corpus_reader(
        words_file=word_dict_srl_file,
        props_file=prop_SRL_file)
  
  c=1
  for x in reader1():
    c+=1
  print c 
  target = open("out/test_data.txt", 'w')
  reader=test();
  counter=0
  for x in reader():
    target.write(str(x))
    target.write(str("\n"))
    counter+=1
  target.close() 
  print 'total test sentences : ', counter
		
		
if __name__ == '__main__':
    main()