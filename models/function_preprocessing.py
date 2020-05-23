import itertools
import re
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import nltk
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score
# nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
# nltk.download('punkt')
# nltk.download("stopwords")
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
import pandas as pd


def clean_data(sentence):
    # removing web links
    s = [re.sub(r'http\S+', '', sentence.lower())]
    # removing words like gooood and poooor to good and poor
    s = [''.join(''.join(s)[:2] for _, s in itertools.groupby(s[0]))]
    # removing appostophes
    s = [remove_appostophes(s[0])]
    # removing punctuations from the code
    s = [remove_punctuations(s[0])]
    return s[0]


def remove_punctuations(my_str):
    punctuations = '''!()-[]{};:'"\,./?@#$%^&@*_~'''
    no_punct = ""
    for char in my_str:
        if char not in punctuations:
            no_punct = no_punct + char
    return no_punct


def remove_appostophes(sentence):
    APPOSTOPHES = {"s": "is", "re": "are", "t": "not",
                   "ll": "will", "d": "had", "ve": "have", "m": "am"}
    words = nltk.tokenize.word_tokenize(sentence)
    final_words = []
    for word in words:
        broken_words = word.split("'")
        for single_words in broken_words:
            final_words.append(single_words)
    reformed = [APPOSTOPHES[word]
                if word in APPOSTOPHES else word for word in final_words]
    reformed = " ".join(reformed)
    return reformed


def preprocessing(path):
    training_dataset = pd.read_csv(path, encoding="ISO-8859-1")
    training_dataset.rename(
        columns={'v1': 'Class', 'v2': 'Text'}, inplace=True)
    training_dataset['numClass'] = training_dataset['Class'].map(
        {'ham': 0, 'spam': 1})
    # print(training_dataset)
    # Renaming columns
    # training_dataset.columns = ["labels", "comment"]
    # Adding a new column to contain target variable
    # training_dataset["b_labels"] = [0 if x == "ham" else 1 for x in training_dataset["labels"]]
    # Y = training_dataset["b_labels"]
    # training_dataset.head()
    # print(Y)

    for index in range(0, len(training_dataset["Text"])):
        training_dataset.loc[index, "Text"] = clean_data(
            training_dataset["Text"].iloc[index])
    output = training_dataset.drop(training_dataset.columns[[2, 3, 4]], axis=1)
    output = output.values.tolist()
    # print(training_dataset["Text"])
    # stopset = set(stopwords.words("english"))
    stopset = set(stopwords.words("english"))

    # Initialising Count Vectorizer
    vectorizer = CountVectorizer(stop_words=stopset, binary=True)
    vectorizer = CountVectorizer()

    X = vectorizer.fit_transform(training_dataset.Text)
    # Extract target column 'Class'
    y = training_dataset.numClass

    # Performing test train Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, train_size=0.70, random_state=None)
    return X_train, X_test, y_train, y_test, output
