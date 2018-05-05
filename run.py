import data
import model
import config
import torch
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F



if __name__ == "__main__":

    print("starting...")
    # prepare data
    csv_dataset = pd.read_csv(config.file_name, header=None)  # csv_file format: dataframe
    print("data loaded")
    vocab = data.Vocabulary()
    data.build_vocab(vocab)  # build vocabulary

    print("build vocab success")
    train_data = data.sentimentDataset(vocab, csv_dataset, train_size=config.TRAIN_RATIO,
                                       test_size=config.TEST_RATIO, train=True)
    test_data = data.sentimentDataset(vocab, csv_dataset, train=False)

    train_dataloader = DataLoader(train_data, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True,
                                  collate_fn=data.collate_fn)
    test_dataloader = DataLoader(test_data, batch_size=config.TEST_BATCH_SIZE, shuffle=True, collate_fn=data.collate_fn)

    model_classifier = model.RNNClassifier(nembedding=config.DIM,
                                           hidden_size=config.HIDDEN_SIZE,
                                           num_layer=config.NUM_LAYER,
                                           dropout=config.drop_out,
                                           vocab_size=vocab.n_words,
                                           use_pretrain=False,
                                           embed_matrix=vocab.vector,
                                           embed_freeze=False,
                                           label_size=2
                                           )

    model_classifier = model.CNNClassifier(nembedding=config.DIM,
                                           vocab_size=vocab.n_words,
                                           kernel_num=3,
                                           kernel_sizes=config.kernel_sizes,
                                           label_size=2,
                                           dropout=0.3,
                                           use_pretrain=False)

    use_loaded_model = input("load model?")
    if use_loaded_model == "yes":
        if isinstance(model_classifier, model.RNNClassifier):
            print("loading RNN...")
            model_classifier.load_state_dict(torch.load("rnn.pkl"))
        if isinstance(model_classifier, model.CNNClassifier):
            print("loading CNN...")
            model_classifier.load_state_dict(torch.load("cnn.pkl", map_location=lambda storage, loc: storage))


    def output_sentiment(sentence):  # sentence is a string
        sentence = sentence.split()  # list
        length = np.array([max(len(sentence), config.kernel_sizes[-1])])  # length must be greater than max kernel size
        sequence = [0] * length[0]
        for i, word in enumerate(sentence):
            sequence[i] = vocab.get_idx(word)

        sequence_tensor = torch.LongTensor(sequence).unsqueeze(1)
        packed_sequences = torch.nn.utils.rnn.pack_padded_sequence(Variable(sequence_tensor).cuda(), length)
        out = model_classifier(packed_sequences)
        out = F.softmax(out, dim=1)
        return out

    if config.use_cuda:
        model_classifier = model_classifier.cuda()



    optimizer = optim.Adam(model_classifier.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()



    def training_update_function(batch):

        optimizer.zero_grad()
        # print(batch)
        x, y = batch['sentence'], batch['sentiment']

        y_pred = model_classifier(x)
        loss = loss_fn(y_pred, y.squeeze())
        loss.backward()
        optimizer.step()
        return loss.data


    def inference_function(batch):
        text, label = batch['sentence'], batch['sentiment']
        out = model_classifier(text)
        _, out_label = torch.max(out, 1)
        accuracy = (torch.sum(out_label == label).cpu().data.numpy() / len(label))  # have to convert bytetensor-->numpy
        # print(type(accuracy))
        return accuracy


    print("training...")
    model_classifier.train()
    for epoch in range(config.EPOCH):
        print("epoch", epoch)
        for batch in train_dataloader:
            print(training_update_function(batch))

    print("saving model")
    if isinstance(model_classifier, model.RNNClassifier):
        torch.save(model_classifier.state_dict(), "rnn.pkl")
    if isinstance(model_classifier, model.CNNClassifier):
        torch.save(model_classifier.state_dict(), "cnn.pkl")

    print("testing...")
    model_classifier.eval()
    sum = 0
    cnt =0
    for batch in test_dataloader:
        sum += inference_function(batch)
        cnt += 1
    print("accuracy:", sum / cnt)
