from .CNN_Humor import CNN
import rl.data_parsing as data_parsing
import pickle
from torch import optim
from torch import nn
from torchtext import data
import numpy as np
import torch
import time
UNK_TOKEN = '<UNK>'

# https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/4%20-%20Convolutional%20Sentiment%20Analysis.ipynb

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


def initialize_with_pretrained(pretrained_embeds, word_embedding, lang, use_cuda=False):
    for word, index in lang.word2index.items():
        if word in pretrained_embeds:
            #print("found word {}".format(word))
            word_embedding.embedding.weight.data[index] = torch.Tensor(pretrained_embeds[word])
        else:
            word_embedding.embedding.weight.data[index] = torch.Tensor(pretrained_embeds[UNK_TOKEN])


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        optimizer.zero_grad()

        predictions = model(batch.sentence).squeeze(1)

        loss = criterion(predictions, batch.score)

        acc = binary_accuracy(predictions, batch.score)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.sentence).squeeze(1)

            loss = criterion(predictions, batch.score)

            acc = binary_accuracy(predictions, batch.score)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def return_cnn_class():
    json_out = data_parsing.parse_reddit_jokes()
    out_lang, pairs = data_parsing.read_langs('reddit_jokes', json_out)
    INPUT_DIM = out_lang.n_words
    MAX_VOCAB_SIZE = 25_000
    train_data, valid_data, test_data, TEXT, LABEL = data_parsing.make_torch_dataset_from_reddit_jokes()
    TEXT.build_vocab(train_data,
                     max_size=MAX_VOCAB_SIZE,
                     vectors="glove.6B.100d",
                     unk_init=torch.Tensor.normal_)
    EMBEDDING_DIM = 100
    N_FILTERS = 10
    FILTER_SIZES = [1, 1, 1, 1]
    OUTPUT_DIM = 1
    DROPOUT = 0.5
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
    model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    return model, TEXT, device

def run_cnn():
    json_out = data_parsing.parse_reddit_jokes()
    out_lang, pairs = data_parsing.read_langs('reddit_jokes', json_out)
    INPUT_DIM = out_lang.n_words
    MAX_VOCAB_SIZE = 25_000
    train_data, valid_data, test_data, TEXT, LABEL = data_parsing.make_torch_dataset_from_reddit_jokes()
    TEXT.build_vocab(train_data,
                     max_size=MAX_VOCAB_SIZE,
                     vectors="glove.6B.100d",
                     unk_init=torch.Tensor.normal_)
    EMBEDDING_DIM = 100
    N_FILTERS = 20
    FILTER_SIZES = [1, 2, 2, 1]
    OUTPUT_DIM = 1
    DROPOUT = 0.5
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
    model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)
    #pret_embs = pickle.load(open('pretrained-embeds-coref.pkl', 'rb'))
    #initialize_with_pretrained(pret_embs, model, out_lang)
    N_EPOCHS = 10
    BATCH_SIZE = 128
    best_valid_loss = float('inf')
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = criterion.to(device)
    # train_ix = int(len(pairs)*0.7)
    # valid_ix = int(len(pairs)*0.2)
    # train_data = pairs[:train_ix]
    # valid_data = pairs[train_ix: train_ix + valid_ix]
    # test_data = pairs[train_ix + valid_ix:]
    #LABEL.build_vocab(train_data)
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=BATCH_SIZE,
        sort_key = lambda x: len(x.sentence),
        sort_within_batch = False,
        device=device)
    print('starting the training thing')
    for epoch in range(N_EPOCHS):
        print('epoch {}'.format(epoch))
        start_time = time.time()

        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tut4-model.pt')

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')
    torch.save(model.state_dict(), 'cnn_filter2.pt')
    #model.load_state_dict(torch.load('tut4-model.pt'))

    test_loss, test_acc = evaluate(model, test_iterator, criterion)

    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')
    return model, TEXT, device


if __name__ == '__main__':
    run_cnn()
