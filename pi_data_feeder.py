"""
Conll03 dataset.
"""

import tarfile
import gzip
import itertools
import collections
import re
import numpy as np

__all__ = ['train', 'test', 'get_dict', 'get_embedding']

UNK_IDX = 0
# unknown token
UNK="UUUNKKK"

train_file = 'data/predicate_identifier/train.txt'
test_file = 'data/predicate_identifier/test.txt'
target_file = 'data/predicate_identifier/target.txt'
vocab_file = 'data/embedding/vocab.txt'
emb_file = 'data/embedding/wordVectors.txt'




def canonicalize_digits(word):
    if any([c.isalpha() for c in word]): return word
    word = re.sub("\d", "DG", word)
    if word.startswith("DG"):
        word = word.replace(",", "")  # remove thousands separator
    return word


def canonicalize_word(word, wordset=None, digits=True):
    word = word.lower()
    if digits:
        if (wordset != None) and (word in wordset): return word
        word = canonicalize_digits(word)  # try to canonicalize numbers
    if (wordset == None) or (word in wordset): return word
    else: return UNK  # unknown token


def load_dict(filename):
    d = dict()
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            d[line.strip()] = i
    return d


def get_dict(vocab_file=vocab_file, target_file=target_file):
    """
    Get the word and label dictionary.
    """
    word_dict = load_dict(vocab_file)
    label_dict = load_dict(target_file)
    print ("Size of vocab :" , len(word_dict))
    print ("total target labels :" , len(label_dict))
    return word_dict, label_dict


def get_embedding(emb_file=emb_file):
    """
    Get the trained word vector.
    """
    return np.loadtxt(emb_file, dtype=float)


def corpus_reader(filename=train_file):
    def reader():
        sentence = []
        labels = []
        with open(filename) as f:
            for line in f:
                if re.match(r"-DOCSTART-.+", line) or (len(line.strip()) == 0):
                    if len(sentence) > 0:
                    	#print sentence,labels
                        yield sentence, labels
                        
                    sentence = []
                    labels = []
                else:
                    segs = line.strip().split()
                    #print segs
                    sentence.append(segs[0])
                    # transform from I-TYPE to BIO schema
                    if segs[-1] != 'O' and (len(labels) == 0 or
                                            labels[-1][1:] != segs[-1][1:]):
                        labels.append('B' + segs[-1][1:])
                    else:
                        labels.append(segs[-1])

        f.close()

    return reader


def reader_creator(corpus_reader, word_dict, label_dict):
    """
    Conll03 train set creator.
    The dataset can be obtained according to http://www.clips.uantwerpen.be/conll2003/ner/.
    It returns a reader creator, each sample in the reader includes word id sequence, label id sequence and raw sentence for purpose of print.
    :return: Training reader creator
    :rtype: callable
    """

    def reader():
        for sentence, labels in corpus_reader():
            word_idx = [
                word_dict.get(canonicalize_word(w, word_dict), UNK_IDX)
                for w in sentence
            ]
            mark = [1 if w[0].isupper() else 0 for w in sentence]
            label_idx = [label_dict.get(w) for w in labels]
            str1 = ' '.join(str(e) for e in sentence)
            #print word_idx, mark, label_idx, str1

            yield word_idx, mark, label_idx, sentence

    return reader



def train(data_file=train_file,
          vocab_file=vocab_file,
          target_file=target_file):
    return reader_creator(
        corpus_reader(data_file),
        word_dict=load_dict(vocab_file),
        label_dict=load_dict(target_file))


def test(data_file=test_file,
         vocab_file=vocab_file,
         target_file=target_file):
    return reader_creator(
        corpus_reader(data_file),
        word_dict=load_dict(vocab_file),
        label_dict=load_dict(target_file))
        

	        
