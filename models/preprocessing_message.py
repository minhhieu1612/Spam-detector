import pickle
# nltk.download('wordnet')
# from nltk.stem import WordNetLemmatizer
# from nltk.corpus import wordnet as wn
# nltk.download('punkt')
# nltk.download("stopwords")
from models import helpers


def predict_preprocessing(sentence):
    data = [helpers.clean_data(sentence)]

    loaded_vectorizer = pickle.load(open('vectorizer.pickle', 'rb'))
    X = loaded_vectorizer.transform(data)

    # Performing test train Split

    return X
