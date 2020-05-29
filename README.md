# SMS-Message-Spam-Detector

A simple Flask API to detect spam or ham using Python and sklearn

#### Members of Team 9:

      - Nguyễn Hữu Định       16520206
      - Nguyễn Đình Cường     16520146
      - Nguyễn Minh Hiếu      16520402
      - Đoàn Đức Đăng Quang   16520989
      - Nguyễn Đình Anh       16520040
      - Lưu Văn Tuấn          16521371

#### Libraries and setup enviroment

- Python package: tensorflow 2.2.0, flask, sklearn, joblib, numpy, pandas, keras,itertools, re, nltk, pickle

- Run project: python app.py

#### Features:

- Read and preprocessing dataset.
- Visualization train and test set.
- Clean data after preprocessing dataset.
- Use 5 model to train dataset: decision tree, naive bayes, k neighbor(KNN), support vector model(SVM), long short-term model(LSTM).
- Predict one or more message(if predict file).
- Compare models through bar chart.
