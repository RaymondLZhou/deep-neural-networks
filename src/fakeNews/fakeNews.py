import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def split(text, labels):
    x_train, x_test, y_train, y_test = train_test_split(text, labels, test_size=0.2, random_state=7)

    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

    tfidf_train = tfidf_vectorizer.fit_transform(x_train) 
    tfidf_test = tfidf_vectorizer.transform(x_test)

    return tfidf_train, y_train, tfidf_test, y_test

def train(tfidf_train, y_train, tfidf_test):
    pac = PassiveAggressiveClassifier(max_iter=50)
    pac.fit(tfidf_train, y_train)

    y_pred = pac.predict(tfidf_test)

    return y_pred

def assess(y_test, y_pred):
    confusionMatrix = confusion_matrix(y_test, y_pred, labels=['REAL', 'FAKE'])
    accuracy = accuracy_score(y_test, y_pred)

    positives = confusionMatrix[0]
    negatives = confusionMatrix[1]

    tp, fn = positives
    fp, tn = negatives

    prec = tp/(tp+fp)
    rec = tp/(tp+fn)

    f1 = 2*prec*rec/(prec+rec)

    print(confusionMatrix)
    print("Accuracy: " + str(round(accuracy, 6)))
    print("Precision: " + str(round(prec, 6)))
    print("Recall: " + str(round(rec, 6)))
    print("F1 Score: " + str(round(f1, 6)))

df = pd.read_csv('../../data/news.csv')
labels = df.label

tfidf_train, y_train, tfidf_test, y_test = split(df['text'], labels)

y_pred = train(tfidf_train, y_train, tfidf_test)

assess(y_test, y_pred)
