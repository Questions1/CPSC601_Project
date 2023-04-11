
from transformers import BertTokenizer, BertForSequenceClassification


if __name__ == '__main__':
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)
    model.config.num_hidden_layers

    num_params = sum(p.numel() for p in model.parameters())

    print("Training data label distribution: -----------\n", train_data['label'].value_counts())
    print("Testing data label distribution: -----------\n", test_data['label'].value_counts())
    print("Training data text max length: -----------\n", train_data['text'].map(len).max())
    print("Training data text min length: -----------\n", train_data['text'].map(len).min())

    wmt = []
    with open('./input/paracrawl.wmt21.en-zh', 'r', encoding='utf-8') as f:
        for i in range(10):
            wmt.append(f.readline())

    with open('./input/paracrawl.wmt21.en-zh', 'r', encoding='utf-8') as f:
        a = f.readlines()

