import torch
import torch.nn as nn
from torchtext.data import Field
from torchtext.data import TabularDataset
import torch.autograd as ag
from torch.autograd import Variable
import torch.cuda as cuda
from torchtext import datasets
import json
import unicodedata
import re

USE_CUDA = False

tokenize = lambda x: x.split()

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def index_words(self, sentence):
        for word in sentence.split(' '):
            self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters
def normalize_string(s):
    #s = unicode_to_ascii(s.lower().strip())
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def read_langs(lang_name, json_out, write_clean_file=False, write_file_type='csv'):
    pairs = []
    print("Reading lines...")
    # Read the file and split into lines
    lines = ['']*len(json_out.keys())

    clean_map_train = {}
    clean_map_test = {}
    clean_map_valid = {}
    train_ix = int(len(json_out.keys())*0.7)
    valid_ix = int(len(json_out.keys())*0.2)
    for i, s in enumerate(json_out.keys()):
        clean_string = normalize_string(s)
        lines[i] = clean_string
        if i < train_ix:
            clean_map_train[clean_string] = float(json_out[s])
        elif i > train_ix and i < train_ix + valid_ix:
            clean_map_valid[clean_string] = float(json_out[s])
        else:
            clean_map_test[clean_string] = float(json_out[s])
        pairs.append((clean_string, json_out[s]))
    if write_clean_file:
        maps = {'train': clean_map_train, 'valid': clean_map_valid, 'test': clean_map_test}
        for f_name in ['train', 'valid', 'test']:
            if write_file_type == 'json':
                with open('reddit_cleaned_{}.json'.format(f_name), 'w') as outfile:
                    json.dump(maps[f_name], outfile)
            elif write_file_type == 'csv':
                lines = [','.join([k , str(maps[f_name][k])]) for k in maps[f_name].keys()]
                doc = '\n'.join(lines)
                with open('reddit_cleaned_{}.csv'.format(f_name), 'w') as outfile:
                    outfile.write(doc)

    #lines = [normalize_string(s) for s in json_out.keys()]
    # Reverse pairs, make Lang instances
    output_lang = Lang(lang_name)
    print("Indexing words...")
    for line in lines:
        output_lang.index_words(line)
    return output_lang, pairs


def parse_reddit_jokes():
    file_name = 'reddit_jokes.json'
    #lines = []
    #scores = []
    map = {}
    with open(file_name, 'r') as json_file:
        data = json.load(json_file)
        for d in data:
            map[d['body']] = float(d['score'])
            #lines.append(d['body'])
            #scores.append(float(d['score']))
    return map


def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

SOS_token = 0
EOS_token = 1


def variable_from_sentence(lang, sentence):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_token)
    var = Variable(torch.LongTensor(indexes).view(-1, 1))
#     print('var =', var)
    if USE_CUDA: var = var.cuda()
    return var


def make_torch_dataset_from_reddit_jokes():
    TEXT = Field(sequential=True, tokenize='spacy', lower=True)
    LABEL = Field(sequential=False, use_vocab=False, dtype=torch.float)
    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
    joke_datafields = [('sentence', TEXT), ('score', LABEL)]
    trn, vld, tst = TabularDataset.splits(path='', train='reddit_cleaned_train.csv',
                                          validation='reddit_cleaned_valid.csv',
                                          test='reddit_cleaned_test.csv',
                                          format='csv',
                                          fields=joke_datafields)
    return trn, vld, tst, TEXT, LABEL


if __name__ == '__main__':
    map = parse_reddit_jokes()
    read_langs('jokes', map, write_clean_file=True, write_file_type='csv')
