from models import preprocessings
from sklearn.externals import joblib


class Predict_message:
    def __init__(self, string):
        self.string = preprocessings.for_message(string)

    def KNN(self):
        modelscorev2 = joblib.load('KNN.pkl', mmap_mode='r')
        prediction = modelscorev2.predict(self.string)
        print('ham' if prediction == 0 else 'spam')
        return prediction

    def DecisionTree(self):
        decisionTree = joblib.load('DecisionTree.pkl', mmap_mode='r')
        prediction = decisionTree.predict(self.string)
        print('ham' if prediction == 0 else 'spam')
        return prediction

    def Naive_bayes(self):
        NB = joblib.load('NB.pkl', mmap_mode='r')
        prediction = NB.predict(self.string)
        print('ham' if prediction == 0 else 'spam')
        return prediction

    def SVM(self):
        svm = joblib.load('SVM.pkl', mmap_mode='r')
        prediction = svm.predict(self.string)
        print('ham' if prediction == 0 else 'spam')
        return prediction

    def Run_All(self):
        return None

# p2 = Predict_message(" i am a student")
# p2.KNN()
# p2.DecisionTree()
# p2.Naive_bayes()
# p2.SVM()
