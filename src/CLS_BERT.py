
import torch
import time
from datetime import timedelta
import numpy as np
import pandas as pd
import transformers
from transformers import BertTokenizer, BertModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from utils import preprocess, concatenate_title_and_description, sample_data
from torch import nn
from CLS_LSTM import TextData, evaluate
from torch.utils.data import DataLoader
from torch import optim


class BertClassifier(nn.Module):
    def __init__(self, freeze_bert: bool):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.linear = nn.Linear(768, 4)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_dict: transformers.tokenization_utils_base.BatchEncoding):
        bert_output = self.bert(**input_dict)
        linear_output = self.linear(bert_output['pooler_output'])
        softmax_output = self.softmax(linear_output)
        return softmax_output


def extract_feature(txt: str) -> np.array:
    """
    Using pretrained bert model to extract the feature of news text as a 768-length vector.
    """
    encoded_input = tokenizer(preprocess(txt), return_tensors='pt')
    output = model(**encoded_input)
    last_cls = output['last_hidden_state'].detach().numpy().squeeze()[0, :]
    return last_cls


def show_distribution():
    """
    Show distribution of sentence length
    """
    import nltk
    import matplotlib.pyplot as plt
    tokens_ = [nltk.tokenize.word_tokenize(x) for x in train_data['Title'] + train_data['Description']]
    plt.figure(figsize=(8, 4))
    plt.hist([len(x) for x in tokens_], bins=50)
    plt.xlabel("Number of tokens in each text")
    plt.ylabel("Count")

    tokens_preprocessed_ = [nltk.tokenize.word_tokenize(preprocess(x)) for x in train_data['Title'] + train_data['Description']]
    plt.figure(figsize=(8, 4))
    plt.hist([len(x) for x in tokens_preprocessed_], bins=50)
    plt.xlabel("Number of tokens in each preprocessed text")
    plt.ylabel("Count")


def train(tokenizer, model, train_loader, loss_fn, optimizer):
    """
    Train model on train_loader with 1 global epoch batch by batch.
    """
    start_time = time.time()
    for batch_data_train, batch_label_train in train_loader:
        encoded_input = tokenizer(batch_data_train, max_length=60, padding='max_length', truncation=True, return_tensors='pt')
        outputs = model(encoded_input)
        loss = loss_fn(outputs, batch_label_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    end_time = time.time()
    return model, end_time-start_time


def evaluate(tokenizer, bert_classifier, test_loader):
    """
    Evaluate the model performance on test_loader.
    """
    start_time = time.time()
    with torch.no_grad():
        y_true = torch.Tensor([])
        y_hat = torch.Tensor([])
        for batch_data_test, batch_label_test in test_loader:
            encoded_input = tokenizer(batch_data_test, max_length=128, padding='max_length', truncation=True, return_tensors='pt')
            batch_outputs = bert_classifier(encoded_input)
            batch_y_hat = torch.argmax(batch_outputs, dim=1)
            y_true = torch.concat([y_true, batch_label_test]).int()
            y_hat = torch.concat([y_hat, batch_y_hat]).int()
        acc = sum(y_true == y_hat) / len(y_hat)
    end_time = time.time()
    return acc, end_time - start_time


def train_evaluate(epoch_num, freeze_bert):
    train_loader = DataLoader(TextData(train_data2['text'], train_data2['label'] - 1), batch_size=128, shuffle=True)
    test_loader = DataLoader(TextData(test_data2['text'], test_data2['label'] - 1), batch_size=1024, shuffle=False)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', bos_token="[CLS]", eos_token="[SEP]")
    bert_classifier = BertClassifier(freeze_bert=freeze_bert)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(bert_classifier.parameters(), lr=0.01)

    total_train_time = 0
    total_evaluate_time = 0
    test_acc_ = []
    for epoch in range(epoch_num):
        bert_classifier, train_time = train(tokenizer, bert_classifier, train_loader, loss_fn, optimizer)
        test_acc, evaluate_time = evaluate(tokenizer, bert_classifier, test_loader)

        total_train_time += train_time
        total_evaluate_time += evaluate_time

        test_acc_.append(float(test_acc))

        print(f"Epoch: {epoch}\t|\tTest Accuracy: {test_acc * 100:.0f}%\t|\t"
              f"TrainTime: {timedelta(seconds=int(total_train_time))}\t|\t"
              f"EvaluateTime: {timedelta(seconds=int(total_evaluate_time))}")


if __name__ == '__main__':
    TRAIN_PATH = './input/AG_NEWS_kaggle/train.csv'
    TEST_PATH = './input/AG_NEWS_kaggle/test.csv'

    train_data = pd.read_csv(TRAIN_PATH)
    test_data = pd.read_csv(TEST_PATH)

    train_data2 = sample_data(concatenate_title_and_description(train_data), 5000)
    test_data2 = sample_data(concatenate_title_and_description(test_data), 1000)

    train_evaluate(30, freeze_bert=False)
    train_evaluate(30, freeze_bert=True)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', bos_token="[CLS]", eos_token="[SEP]")
    model = BertModel.from_pretrained("bert-base-uncased")

    # train_x = []
    # start_time = time.time()
    # for i, text in enumerate(train_data2['text']):
    #     train_x.append(extract_feature(text))
    #     if (i % 1000 == 0) & (i > 0):
    #         time_gap = time.time()-start_time
    #         print(i, '\t', timedelta(seconds=int(time_gap)), f"Anticipated Total Time: {timedelta(seconds=int(time_gap * len(train_data2) / i))}")
    #
    # train_x = train_data2['text'].apply(extract_feature)
    # test_x = test_data2['text'].apply(extract_feature)
    #
    # train_y = train_data2['label'].values
    # test_y = test_data2['label'].values
    #
    # start_time = time.time()
    # cls_model = RandomForestClassifier()
    # cls_model.fit(pd.DataFrame([x for x in train_x]), train_y)
    # print(f"Train Time: {timedelta(seconds=int(time.time()-start_time))}")
    # y_hat = cls_model.predict(pd.DataFrame([x for x in test_x]))
    # acc = accuracy_score(test_y, y_hat)
    # print(f"Test ACC: {acc * 100:.2f}%")

