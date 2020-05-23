from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pickle
# nltk.download('wordnet')
# from nltk.stem import WordNetLemmatizer
# from nltk.corpus import wordnet as wn
# nltk.download('punkt')
# nltk.download("stopwords")
from nltk.corpus import stopwords
import pandas as pd
from models import helpers


def preprocessing(path):
    training_dataset = pd.read_csv(path, encoding="ISO-8859-1")
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
