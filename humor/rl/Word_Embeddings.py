import torch
import torch.nn as nn
import torch.autograd as ag
import torch.nn.functional as F
import torch.cuda as cuda


class BiLSTMWordEmbedding(nn.Module):
    '''
    In this component, you will use a Bi-Directional LSTM to get initial embeddings.
    The embedding for word i is the i'th hidden state of the LSTM
    after passing the sentence through the LSTM.
    '''

    ## deliverable 4.1
    def __init__(self, word_to_ix, word_embedding_dim, hidden_dim, num_layers, dropout):
        '''
        :param word_to_ix: dict mapping words to unique indices
        :param word_embedding_dim: the dimensionality of the input word embeddings
        :param hidden_dim: the dimensionality of the output embeddings that go to the classifier
        :param num_layers: the number of LSTM layers to use
        :param dropout: amount of dropout in LSTM
        '''
        super(BiLSTMWordEmbedding, self).__init__()
        self.word_to_ix = word_to_ix
        self.num_layers = num_layers
        self.word_embedding_dim = word_embedding_dim
        self.hidden_dim = hidden_dim
        self.use_cuda = False

        self.output_dim = hidden_dim

        # STUDENT
        # Construct the needed components in this order:
        # 1. An embedding lookup table
        # 2. The LSTM
        # Note we want the output dim to be hidden_dim, but since our LSTM
        # is bidirectional, we need to make the output of each direction hidden_dim/2
        # name your embedding member "word_embeddings"
        self.word_embeddings = nn.Embedding(num_embeddings=len(self.word_to_ix), embedding_dim=self.word_embedding_dim)
        self.lstm = nn.LSTM(input_size=self.word_embedding_dim, hidden_size=self.hidden_dim // 2,
                            num_layers=self.num_layers, dropout=dropout, bidirectional=True)
        # END STUDENT

        self.hidden = self.init_hidden()

    ## deliverable 4.1
    def forward(self, document):
        '''
        This function has several parts.
        1. Look up the embeddings for the words in the document.
           These will be the inputs to the LSTM sequence model.
           NOTE: At this step, rather than a list of embeddings, it should be a single tensor.
        2. Now that you have your tensor of embeddings, You can pass it through your LSTM.
        3. Convert the outputs into the correct return type, which is a list of lists of
           embeddings, each of shape (1, hidden_dim)
        NOTE: Make sure you are reassigning self.hidden to the new hidden state!
        :param document: a list of strs, the words of the document
        :returns: a list of embeddings for the document
        '''
        assert self.word_to_ix is not None, "ERROR: Make sure to set word_to_ix on \
                the embedding lookup components"
        # STUDENT
        storage = ag.Variable(torch.FloatTensor(len(document), 1, self.word_embedding_dim))
        for i, word in enumerate(document):
            wi = self.word_to_ix[word]
            lookup_thing = ag.Variable(torch.LongTensor([wi]))
            embed = self.word_embeddings(lookup_thing)
            storage[i, 0, :] = embed
        embeds, self.hidden = self.lstm(storage, self.hidden)
        return [embed for embed in embeds]
        # END STUDENT

    def init_hidden(self):
        '''
        PyTorch wants you to supply the last hidden state at each timestep
        to the LSTM.  You shouldn't need to call this function explicitly
        '''
        if self.use_cuda:
            return (ag.Variable(cuda.FloatTensor(self.num_layers * 2, 1, self.hidden_dim//2).zero_()),
                    ag.Variable(cuda.FloatTensor(self.num_layers * 2, 1, self.hidden_dim//2).zero_()))
        else:
            return (ag.Variable(torch.zeros(self.num_layers * 2, 1, self.hidden_dim//2)),
                    ag.Variable(torch.zeros(self.num_layers * 2, 1, self.hidden_dim//2)))

    def clear_hidden_state(self):
        self.hidden = self.init_hidden()

    def to_cuda(self):
        self.use_cuda = True
        self.cuda()