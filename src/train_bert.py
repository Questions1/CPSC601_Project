
import pandas as pd
from transformers import BertModel
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from CLS_BERT import concatenate_title_and_description
from sklearn.metrics import accuracy_score


if __name__ == '__main__':
    # bert = BertModel.from_pretrained('bert-base-uncased')
    # num_layers = bert.config.num_hidden_layers

    # Import the required libraries
    from sklearn.metrics import accuracy_score

    # Load the data
    TRAIN_PATH = './input/AG_NEWS_kaggle/train.csv'
    TEST_PATH = './input/AG_NEWS_kaggle/test.csv'

    train_data = pd.read_csv(TRAIN_PATH)
    test_data = pd.read_csv(TEST_PATH)

    train_data = concatenate_title_and_description(train_data)
    test_data = concatenate_title_and_description(test_data)

    # Tokenize the data
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_encodings = tokenizer(list(train_data['text']), truncation=True, padding=True)
    test_encodings = tokenizer(list(test_data['text']), truncation=True, padding=True)

    # Encode the data
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_encodings['input_ids']),
                                                   torch.tensor(train_encodings['attention_mask']),
                                                   torch.tensor(train_data['label']))
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(test_encodings['input_ids']),
                                                  torch.tensor(test_encodings['attention_mask']),
                                                  torch.tensor(test_data['label']))

    # Define the model
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)

    # Train the model
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    for epoch in range(3):
        for batch in train_loader:
            optimizer.zero_grad()
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

    # Evaluate the model
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1]}
            outputs = model(**inputs)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=1)
            predictions.extend(pred.tolist())

    # Compute the accuracy
    true_labels = test_data['label'].tolist()
    accuracy = accuracy_score(true_labels, predictions)
    print(f"Accuracy: {accuracy}")

