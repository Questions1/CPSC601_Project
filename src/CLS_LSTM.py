
import time
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
from datetime import timedelta
from utils import concatenate_title_and_description, sample_data, preprocess
import torchtext
from torch.utils.data import Dataset, DataLoader
from torch import optim
import matplotlib.pyplot as plt
from tqdm import tqdm


# class LSTMClassifier(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
#         super(LSTMClassifier, self).__init__()
#         self.num_layers = num_layers
#         self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
#         self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
#         self.linear = nn.Linear(hidden_dim, 4)
#
#     def forward(self, X_batch):
#         embeddings = self.embedding_layer(X_batch)
#         hidden, carry = torch.randn(self.num_layers, len(X_batch), hidden_dim), torch.randn(self.num_layers, len(X_batch), hidden_dim)
#         output, (hidden, carry) = self.lstm(embeddings, (hidden, carry))
#         return self.linear(output[:, -1])


class LSTM(nn.Module):
    """
    This is a versatile LSTM model with being able to initialize three kinds of LSTM models:
    1: LSTM without pretrained embeddings.
    2: LSTM with pretrained embeddings, which will be fine-tuned during training.
    3: LSTM with pretrained embeddings, which will NOT be fine-tuned during training.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers,
                 pretrained_embedding=None, fine_tune_embedding=False):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        if pretrained_embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding, freeze=fine_tune_embedding)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 4)

    def forward(self, X_batch):
        embeddings = self.embedding(X_batch)
        # hidden, carry = torch.randn(self.num_layers, len(X_batch), hidden_dim), torch.randn(self.num_layers, len(X_batch), hidden_dim)
        lstm_output, (h_n, c_n) = self.lstm(embeddings)
        output = self.linear(lstm_output[:, -1, :])
        return output


class TextData(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def get_data_loader(data, word_to_idx, max_len, prep=True, batch_size=64):
    """
    Obtain the train_loader and test_loader
    """
    seq = []
    for i, text in enumerate(data['text']):
        if prep:
            tokens = tokenizer(preprocess(text))
        else:
            tokens = tokenizer(text)
        indices = [word_to_idx[token] if token in word_to_idx.keys() else 1 for token in tokens]
        if len(indices) < max_len:
            indices = indices + [0] * (max_len - len(indices))
        else:
            indices = indices[:max_len]
        tensor = torch.LongTensor(indices)
        seq.append(tensor)
    data_loader = DataLoader(TextData(seq, data['label']-1), batch_size=batch_size, shuffle=True)
    return data_loader


def train(model, train_loader, loss_fn, optimizer):
    """
    Train model on train_loader with 1 global epoch batch by batch.
    """
    start_time = time.time()
    for batch_data_train, batch_label_train in train_loader:
        outputs = model(batch_data_train)
        loss = loss_fn(outputs, batch_label_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    end_time = time.time()
    return model, end_time-start_time


def evaluate(model, test_loader):
    """
    Evaluate the model performance on test_loader.
    """
    start_time = time.time()
    with torch.no_grad():
        y_true = torch.Tensor([])
        y_hat = torch.Tensor([])
        for batch_data_test, batch_label_test in test_loader:
            batch_outputs = model(batch_data_test)
            batch_y_hat = torch.argmax(batch_outputs, dim=1)
            y_true = torch.concat([y_true, batch_label_test]).int()
            y_hat = torch.concat([y_hat, batch_y_hat]).int()
        acc = sum(y_true == y_hat) / len(y_hat)
    end_time = time.time()
    return acc, end_time-start_time


def load_pretrained_embedding(embedding_dim=50):
    word_to_idx = {'<pad>': 0, '<unk>': 1}
    idx_to_word = ['<pad>', '<unk>']
    embedding_value = [np.zeros(embedding_dim), np.ones(embedding_dim)]
    with open("./input/WordEmbedding/glove.6B/glove.6B.50d.txt", 'r') as f:
        for i, line in enumerate(f):
            # split word and embedding vector
            values = line.split()
            word = values[0]
            embedding_vector = np.array(values[1:]).astype("float32")
            # save word and index to dict, save embedding vector to list container
            word_to_idx[word] = i
            idx_to_word.append(word)
            embedding_value.append(embedding_vector)
    embedding_value = torch.Tensor(embedding_value)
    return word_to_idx, idx_to_word, embedding_value


def check_distribution(train_data):
    """
    Check the distribution of sequence length before and after preprocessing.
    """



def compare_preprocessing():
    test_acc = {True: [], False: []}
    for prep in [True, False]:
        use_pretrained_embedding = False
        tune_pretrained_embedding = False
        if use_pretrained_embedding:  # Train embedding from zero
            word_to_idx, idx_to_word, embedding_value = load_pretrained_embedding()
            model = LSTM(len(word_to_idx), embedding_dim, hidden_dim, num_layers, pretrained_embedding=embedding_value,
                         fine_tune_embedding=tune_pretrained_embedding)
        else:
            vocab = torchtext.vocab.build_vocab_from_iterator(map(tokenizer, train_data2['text']), min_freq=1,
                                                              specials=['<pad>', '<unk>'])
            vocab.set_default_index(vocab['<unk>'])
            word_to_idx = vocab.get_stoi()
            idx_to_word = vocab.get_itos()
            model = LSTM(len(word_to_idx), embedding_dim, hidden_dim, num_layers)

        # Generate train_loader and test_loader
        train_loader = get_data_loader(train_data2, word_to_idx, max_len=60 - prep * 25, prep=prep, batch_size=batch_size)
        test_loader = get_data_loader(test_data2, word_to_idx, max_len=60 - prep * 25, prep=prep, batch_size=batch_size)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        total_train_time = 0
        total_evaluate_time = 0
        total_time = 0

        # Start training...
        for epoch in range(num_epochs):
            model, train_time = train(model, train_loader, loss_fn, optimizer)
            acc, evaluate_time = evaluate(model, test_loader)

            total_train_time += train_time
            total_evaluate_time += evaluate_time

            test_acc[prep].append(acc)

            print(f"With preprocessing: {prep}\t|\tEpoch: {epoch}\t|\tTest Accuracy: {acc * 100:.0f}%\t|\t"
                  f"TrainTime: {timedelta(seconds=int(total_train_time))}\t|\t"
                  f"EvaluateTime: {timedelta(seconds=int(total_evaluate_time))}")
    plt.figure(figsize=(8, 4))
    plt.plot(range(len(test_acc[True])), test_acc[True], label='with preprocessing')
    plt.plot(range(len(test_acc[False])), test_acc[False], label='without preprocessing')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.savefig('./preprocessing_effect.png')
    plt.close()
    return test_acc


def compare_embedding():
    test_acc = [[], [], []]
    for i, (use_pretrained_embedding, tune_pretrained_embedding) in enumerate([(False, None), (True, False), (True, True)]):
        if use_pretrained_embedding:  # Train embedding from zero
            word_to_idx, idx_to_word, embedding_value = load_pretrained_embedding()
            model = LSTM(len(word_to_idx), embedding_dim, hidden_dim, num_layers, pretrained_embedding=embedding_value,
                         fine_tune_embedding=tune_pretrained_embedding)
        else:
            vocab = torchtext.vocab.build_vocab_from_iterator(map(tokenizer, train_data2['text']), min_freq=1,
                                                              specials=['<pad>', '<unk>'])
            vocab.set_default_index(vocab['<unk>'])
            word_to_idx = vocab.get_stoi()
            idx_to_word = vocab.get_itos()
            model = LSTM(len(word_to_idx), embedding_dim, hidden_dim, num_layers)

        # Generate train_loader and test_loader
        train_loader = get_data_loader(train_data2, word_to_idx, max_len=35, prep=True, batch_size=batch_size)
        test_loader = get_data_loader(test_data2, word_to_idx, max_len=35, prep=True, batch_size=batch_size)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        total_train_time = 0
        total_evaluate_time = 0
        total_time = 0

        # Start training...
        for epoch in range(num_epochs):
            model, train_time = train(model, train_loader, loss_fn, optimizer)
            acc, evaluate_time = evaluate(model, test_loader)

            total_train_time += train_time
            total_evaluate_time += evaluate_time

            test_acc[i].append(acc)

            print(f"Embedding Type: {i}\t|\tEpoch: {epoch}\t|\tTest Accuracy: {acc * 100:.0f}%\t|\t"
                  f"TrainTime: {timedelta(seconds=int(total_train_time))}\t|\t"
                  f"EvaluateTime: {timedelta(seconds=int(total_evaluate_time))}")
    plt.figure(figsize=(8, 4))
    plt.plot(range(len(test_acc[0])), test_acc[0], label='without pretrained embedding')
    plt.plot(range(len(test_acc[1])), test_acc[1], label='with fixed pretrained embedding')
    plt.plot(range(len(test_acc[2])), test_acc[2], label='fine-tune pretrained embedding')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.savefig('./embedding_effect_2.png')
    plt.close()
    return test_acc


if __name__ == '__main__':
    # Load data into memory and concatenate 'title' and 'description'.
    TRAIN_PATH = './input/AG_NEWS_kaggle/train.csv'
    TEST_PATH = './input/AG_NEWS_kaggle/test.csv'
    train_data = pd.read_csv(TRAIN_PATH)
    test_data = pd.read_csv(TEST_PATH)
    train_data2 = sample_data(concatenate_title_and_description(train_data), 5000)
    test_data2 = sample_data(concatenate_title_and_description(test_data), 1000)

    # Obtain the vocabulary from train_data2
    tokenizer = torchtext.data.utils.get_tokenizer('basic_english')

    # Configure some hyper-parameters
    max_len = 35
    embedding_dim = 50
    hidden_dim = 64
    num_layers = 2
    num_epochs = 100
    batch_size = 1024

    # compare_preprocessing()
    compare_embedding()



