import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
import csv
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SequentialSampler
from torch.autograd import Variable
import config
import re
import gensim


def collate_fn(batch):  # rewrite collate_fn to form a mini-batch
    lengths = np.array([len(data['sentence']) for data in batch])
    sorted_index = np.argsort(-lengths)
    lengths = lengths[sorted_index]  # descend order

    max_length = lengths[0]
    batch_size = len(batch)
    sentence_tensor = torch.LongTensor(batch_size, int(max_length)).zero_()

    for i, index in enumerate(sorted_index):
        sentence_tensor[i][:lengths[i]] = torch.LongTensor(batch[index]['sentence'][:max_length])

    sentiments = torch.autograd.Variable(torch.LongTensor([batch[i]['sentiment'] for i in sorted_index]))
    if config.use_cuda:
        packed_sequences = torch.nn.utils.rnn.pack_padded_sequence(Variable(sentence_tensor.t()).cuda(),
                                                                   lengths)  # remember to transpose
        sentiments = sentiments.cuda()
    else:
        packed_sequences = torch.nn.utils.rnn.pack_padded_sequence(Variable(sentence_tensor.t()),
                                                                   lengths)  # remember to transpose
    return {'sentence': packed_sequences, 'sentiment': sentiments}


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9()!?\'\`]", "", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'m", " \'m", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\@.*?[\s\n]", "", string)
    string = re.sub(r"https*://.+[\s]", "", string)
    return string.strip().lower()


class sentimentDataset(Dataset):
    def __init__(self, vocab, dataset, train_size=0.7, test_size=0.3, train=True):  # dataset
        self.train = train
        train_data, test_data = train_test_split(dataset, train_size=train_size, test_size=test_size,
                                                 shuffle=True)  # todo:change
        if self.train:
            self.dataset = train_data
        else:
            self.dataset = test_data

        self.sentence = self.dataset[:][0].tolist()
        self.sentiment = self.dataset[:][1].tolist()
        self.vocab = vocab

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data_idx = {'sentence': [], 'sentiment': self.sentiment[idx]}
        for i, word in enumerate(self.sentence[idx].split()):
            data_idx['sentence'].append(self.vocab.get_idx(word))

        return data_idx  # return index rather than word

    def test(self, idx):
        return self.__getitem__(idx)


def build_vocab(vocab):
    word_vector = gensim.models.KeyedVectors.load_word2vec_format(
        fname='GoogleNews-vectors-negative300-SLIM.bin', binary=True)
    words = word_vector.vocab
    vocab.add_word('<pad>')
    vocab.add_word('<unk>')
    unk_vector = np.random.uniform(-0.25, 0.25, size=config.DIM)  # unk random initial
    vocab.vector = np.zeros((config.MAX_VOCAB_SIZE, config.DIM), dtype=np.float)
    vocab.vector[1][:] = unk_vector
    for word in words:
        vocab.vector[vocab.n_words][:] = word_vector[word]
        vocab.add_word(word)
        if vocab.n_words == config.MAX_VOCAB_SIZE:
            break









class Vocabulary():
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.n_words = 0
        self.vector = ""

    def get_idx(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def get_word(self, idx):
        if idx >= self.n_words:
            print("index out of range")
            return None
        else:
            return self.idx2word[idx]

    def add_word(self, word):
        self.word2idx[word] = self.n_words
        self.idx2word[self.n_words] = word
        self.n_words += 1
