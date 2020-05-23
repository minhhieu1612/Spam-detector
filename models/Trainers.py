from models import Measure
from models import preprocessing_data
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn import feature_extraction, model_selection, naive_bayes, metrics, svm
# from sklearn.metrics import f1_score, accuracy_score
# from sklearn import metrics, feature_extraction
# from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.externals import joblib

dtc = DecisionTreeClassifier(min_samples_split=7, random_state=None)
# huấn luyện mô hình bằng tập train và test


def DecisionTree(X_train, X_test, y_train, y_test):
    dtc.fit(X_train, y_train)
    # dự đoán cho tập dữ liệu test
    y_dtc = dtc.predict(X_test)
    cm = confusion_matrix(y_dtc, y_test)
    print(cm)
    print('Decision Tree Accuracy: ', accuracy_score(y_test, y_dtc))
    # in ra độ chính xác của các label
    print(classification_report(y_test, dtc.predict(X_test)))
    joblib.dump(dtc, 'DecisionTree.pkl')
    precision, recall, fscore, support = score(
        y_test, y_dtc, average='weighted')
    acc_score = accuracy_score(y_test, y_dtc)
    # print('Decision Tree Accuracy: ', accuracy_score(y_test, y_dtc))
    # print('Precision : {}'.format(precision))
    # print('Recall    : {}'.format(recall))
    # print('F-score   : {}'.format(fscore))
    # print('Support   : {}'.format(support))
    return Measure.Measure(acc_score, precision, recall, fscore)


def Naive_Bayes(X_train, X_test, y_train, y_test):
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    y_mnb = clf.predict(X_test)
    cm = confusion_matrix(y_mnb, y_test)
    print(cm)
    print('Naive Bayes Accuracy: ', accuracy_score(y_test, y_mnb))
    # #in ra độ chính xác của các label
    print(classification_report(y_test, clf.predict(X_test)))
    joblib.dump(clf, 'NB.pkl')
    precision, recall, fscore, support = score(
        y_test, y_mnb, average='weighted')
    acc_score = accuracy_score(y_test, y_mnb)
    # print('Decision Tree Accuracy: ', accuracy_score(y_test, y_dtc))
    # print('Precision : {}'.format(precision))
    # print('Recall    : {}'.format(recall))
    # print('F-score   : {}'.format(fscore))
    # print('Support   : {}'.format(support))
    return Measure.Measure(acc_score, precision, recall, fscore)


def KNN(X_train, X_test, y_train, y_test):
    KNN = KNeighborsClassifier(n_neighbors=1)
    KNN.fit(X_train, y_train)
    y_knc = KNN.predict(X_test)
    print('KNeighbors Accuracy_score: ', accuracy_score(y_test, y_knc))
    print('KNeighbors confusion_matrix:/n', confusion_matrix(y_test, y_knc))
    print(classification_report(y_test, KNN.predict(X_test)))
    joblib.dump(KNN, 'KNN.pkl')
    precision, recall, fscore, support = score(
        y_test, y_knc, average='weighted')
    acc_score = accuracy_score(y_test, y_knc)
    # print('Decision Tree Accuracy: ', accuracy_score(y_test, y_dtc))
    # print('Precision : {}'.format(precision))
    # print('Recall    : {}'.format(recall))
    # print('F-score   : {}'.format(fscore))
    # print('Support   : {}'.format(support))
    return Measure.Measure(acc_score, precision, recall, fscore)


def SVM(X_train, X_test, y_train, y_test):
    SVM = svm.SVC(kernel='linear')  # Linear Kernel
    # Train the model using the training sets
    SVM.fit(X_train, y_train)
    # Predict the response for test dataset
    y_pred = SVM.predict(X_test)
    cm = confusion_matrix(y_pred, y_test)
    print(cm)
    print('svm  Accuracy: ', accuracy_score(y_test, y_pred))
    # #in ra độ chính xác của các label
    print(classification_report(y_test, SVM.predict(X_test)))
    joblib.dump(SVM, 'SVM.pkl')
    precision, recall, fscore, support = score(
        y_test, y_pred, average='weighted')
    acc_score = accuracy_score(y_test, y_pred)
    # print('Decision Tree Accuracy: ', accuracy_score(y_test, y_dtc))
    # print('Precision : {}'.format(precision))
    # print('Recall    : {}'.format(recall))
    # print('F-score   : {}'.format(fscore))
    # print('Support   : {}'.format(support))
    return Measure.Measure(acc_score, precision, recall, fscore)


class Trainers:
    def __init__(self, path):
        self.path = path

    def KNN(self):
        X_train, X_test, y_train, y_test, output = preprocessing_data.preprocessing(
            self.path)
        return KNN(X_train, X_test, y_train, y_test)

    def DecisionTree(self):
        X_train, X_test, y_train, y_test, output = preprocessing_data.preprocessing(
            self.path)
        return DecisionTree(X_train, X_test, y_train, y_test)

    def Naive_bayes(self):
        X_train, X_test, y_train, y_test, output = preprocessing_data.preprocessing(
            self.path)
        return Naive_Bayes(X_train, X_test, y_train, y_test)

    def SVM(self):
        X_train, X_test, y_train, y_test, output = preprocessing_data.preprocessing(
            self.path)
        return SVM(X_train, X_test, y_train, y_test)

    def Run_All(self):
        return [
            {
                'trainer': 'KNN',
                'result': self.KNN().getObj()
            },
            {
                'trainer': 'DecisionTree',
                'result': self.DecisionTree().getObj()
            },
            {
                'trainer': 'Naive_bayes',
                'result': self.Naive_bayes().getObj()
            },
            {
                'trainer': 'SVM',
                'result': self.SVM().getObj()
            },
        ]


# p1 = Trainers("spam.csv")
# p1=SMS_spam_detect()

# p1.SVM_predict("I'm gonna be home soon and i don't want to talk about this https://stackoverflow.com/questions/44193154/notfittederror-tfidfvectorizer-vocabulary-wasnt-fitted stuff anymore tonight, k? I've cried enough today. ")
# p1.DecisionTree()
