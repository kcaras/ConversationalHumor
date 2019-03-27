import sklearn
import json
import pickle
import torch
import numpy as np
UNK_TOKEN = '<UNK>'



def obtain_polyglot_embeddings(filename, word_to_ix):
    vecs = pickle.load(open(filename, 'rb'), encoding='latin1')

    vocab = [k for k, v in word_to_ix.items()]

    word_vecs = {}
    for i, word in enumerate(vecs[0]):
        if word in word_to_ix:
            word_vecs[word] = np.array(vecs[1][i])

    word_embeddings = []
    for word in vocab:
        if word in word_vecs:
            embed = torch.Tensor(word_vecs[word])
        else:
            embed = torch.Tensor(word_vecs[UNK_TOKEN])
        word_embeddings.append(embed)

    word_embeddings = np.array(word_embeddings)
    return word_embeddings

def initialize_with_pretrained(pretrained_embeds, word_embedding, use_cuda=False):
    '''
    Initialize the embedding lookup table of word_embedding with the embeddings
    from pretrained_embeds.
    Remember that word_embedding has a word_to_ix member you will have to use.
    For every word that we do not have a pretrained embedding for, keep the default initialization.
    NOTE: don't forget the UNK token!
    :param pretrained_embeds: dict mapping word to python list of floats (the embedding
        of that word)
    :param word_embedding: The network component to initialize (i.e, a BiLSTMWordEmbedding)
    '''
    for word, index in word_embedding.word_to_ix.items():
        if word in pretrained_embeds:
            word_embedding.word_embeddings.weight.data[index] = torch.Tensor(pretrained_embeds[word])
        else:
            word_embedding.word_embeddings.weight.data[index] = torch.Tensor(pretrained_embeds[UNK_TOKEN])


def parse_reddit_jokes():
    file_name = 'reddit_jokes.json'
    lines = []
    scores = []
    with open(file_name, 'r') as json_file:
        data = json.load(json_file)
        for d in data:
            lines.append(d['body'])
            scores.append(float(d['score']))
    return lines, scores


def train_embedder():
    embedding_dim = 30
    hidden_dim = 30
    model = bilstm.BiLSTM(len(word_to_ix), tag_to_ix, embedding_dim, hidden_dim)


def train_scorer():
    pass


def run():
    pass