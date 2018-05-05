import torch
import torch.nn as nn
import torch.nn.functional as F
import config
import numpy as np

from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# pack_padded_sequence：Packs a Variable containing padded sequences of variable length，must be decrease order
# pad_packed_sequence:inverse op of pack_padded_sequence,return the padded sequence and a list of lengths


class RNNClassifier(nn.Module):
    def __init__(self, nembedding, hidden_size, num_layer, dropout,
                 vocab_size, label_size, use_pretrain=False, embed_matrix=None, embed_freeze=True):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, nembedding)
        if use_pretrain is True:
            self.embedding.weight = nn.Parameter(torch.from_numpy(embed_matrix).type(torch.FloatTensor),
                                                 requires_grad=not embed_freeze)

        self.gru = nn.GRU(input_size=nembedding,
                          hidden_size=hidden_size,
                          num_layers=num_layer,
                          dropout=dropout,
                          bidirectional=False)
        self.dense = nn.Linear(in_features=hidden_size,
                               out_features=label_size)

    def forward(self, sequences):
        padded_sentences, lengths = pad_packed_sequence(sequences, padding_value=int(0))
        embeds = self.embedding(padded_sentences)
        packed_embeds = pack_padded_sequence(embeds, lengths)
        out, _ = self.gru(packed_embeds)
        out, lengths = pad_packed_sequence(out, batch_first=False)
        lengths = [l - 1 for l in lengths]
        last_output = out[lengths, range(len(lengths))]
        logits = self.dense(last_output)
        return logits


class CNNClassifier(nn.Module):
    def __init__(self, nembedding, vocab_size, kernel_num, kernel_sizes, label_size,
                 dropout=0.3, use_pretrain=False, embed_matrix=None, embed_freeze=False):
        super(CNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, nembedding)
        if use_pretrain is True:
            self.embedding.weight = nn.Parameter(torch.from_numpy(embed_matrix).type(torch.FloatTensor),
                                                 requires_grad=not embed_freeze)
        self.in_channel = 1
        self.out_channel = kernel_num
        self.kernel_sizes = kernel_sizes
        self.kernel_num = kernel_num
        self.convs1 = nn.ModuleList([nn.Conv2d(self.in_channel, self.out_channel, (K, nembedding))
                                     for K in self.kernel_sizes])  # kernel_sizes,like (3,4,5)

        self.dropout = nn.Dropout(dropout)
        """
        in_feature=len(kernel_sizes)*kernel_num,because we concatenate 
        """
        self.fc = nn.Linear(len(kernel_sizes) * kernel_num, label_size)

    def forward(self, sequences):
        padded_sentences, lengths = pad_packed_sequence(sequences, padding_value=int(0),
                                                        batch_first=True)  # set batch_first true
        x = self.embedding(padded_sentences)  # batch_size*num_word*nembedding

        x = x.unsqueeze(1)  # (batch_size,1,num_word,nembedding)   1 is in_channel

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # a list containing (batch_size,out_channel,W)

        x = [F.max_pool1d(e, e.size(2)).squeeze(2) for e in
             x]  # max_pool1d(input, kernel_size),now x is a list of (batch_size,out_channel)

        x = torch.cat(x, dim=1)  # concatenate , x is batch_size,len(kernel_sizes)*kernel_num

        x = self.dropout(x)
        logits = self.fc(x)

        return logits
