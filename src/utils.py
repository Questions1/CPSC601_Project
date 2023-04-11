
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
import re


def preprocess(sentence: str):
    """
    The first preprocessing step is removing all punctuation and digits to removing useless information.
    The second preprocessing step is transform all words to lower case, since the case information is less useful in the doc2bow method.
    The third step is tokenizing the sentence to prepare for further preprocessing in word level.
    The fourth step is removing stopwords, preposition and subordinating conjunction (IN), cardinal number (CD), modal (MD) words.
    The fifth step is the lemmatization of the word to align the form of same words.
    """
    sentence = re.sub(r'[^a-zA-Z_\s]', '', sentence)  # remove all punctuation and digits
    sentence = sentence.lower()  # lower the case
    tokens = word_tokenize(sentence)  # tokenize the sentence
    tagged = pos_tag(tokens)
    tokenized = [token for token, pos in tagged if token.isalpha() and pos not in {"IN", "CD", "MD"} and token not in stopwords.words('english')]
    lemma = WordNetLemmatizer()
    normalized = [lemma.lemmatize(word) for word in tokenized]
    return ' '.join(normalized)


def concatenate_title_and_description(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    In the original dataset, the news title and description are in two different columns.
    And I am going to concatenate them together to form a news text for later processing.
    """
    df_tmp = dataframe.copy()
    df_tmp['text'] = df_tmp['Title'] + df_tmp['Description']
    df_tmp.drop(columns=['Title', 'Description'], inplace=True)
    df_tmp.rename(columns={'Class Index': 'label'}, inplace=True)
    return df_tmp


def sample_data(data, num):
    idx = np.random.choice(range(len(data)), num, replace=False)
    return data.iloc[idx, :].reset_index(drop=True)
