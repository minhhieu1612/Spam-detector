from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import pickle
from models import helpers


def for_message(message):
    data = [helpers.clean_data(message)]

    loaded_vectorizer = pickle.load(open('vectorizer.pickle', 'rb'))
    X = loaded_vectorizer.transform(data)

    # Performing test train Split

    return X


def for_dataset(path_file):
    training_dataset = pd.read_csv(path_file, encoding="ISO-8859-1")
    # print(training_dataset.head(10))
    training_dataset.drop(
        ["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1, inplace=True)
    training_dataset.rename(
        columns={'v1': 'Class', 'v2': 'Text'}, inplace=True)
    training_dataset['numClass'] = training_dataset['Class'].map(
        {'ham': 0, 'spam': 1})

    for index in range(0, len(training_dataset["Text"])):
        training_dataset.loc[index, "Text"] = helpers.clean_data(
            training_dataset["Text"].iloc[index])

    output = training_dataset
    # .drop(training_dataset.columns[[2, 3, 4]], axis=1)
    output = output.values.tolist()
    # print(training_dataset.head(10))
    # print(training_dataset["Text"])
    # stopset = set(stopwords.words("english"))
    stopset = set(stopwords.words("english"))

    # Initialising Count Vectorizer
    vectorizer = CountVectorizer(stop_words=stopset, binary=True)
    vectorizer = CountVectorizer()

    X = vectorizer.fit_transform(training_dataset.Text)
    # Extract target column 'Class'
    y = training_dataset.numClass
    vec_file = 'vectorizer.pickle'
    pickle.dump(vectorizer, open(vec_file, 'wb'))
    # Performing test train Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, train_size=0.70, random_state=None)
    return X_train, X_test, y_train, y_test, output


def for_file(path_file):
    data_test = pd.read_csv(path_file, encoding="ISO-8859-1")
    print(data_test.head(10))
    for index in range(0, len(data_test["message"])):
        data_test.loc[index, "message"] = helpers.clean_data(
            data_test["message"].iloc[index])

    # Initialising Count Vectorizer
    loaded_vectorizer = pickle.load(open('vectorizer.pickle', 'rb'))
    X = loaded_vectorizer.transform(data_test["message"])
    return X
