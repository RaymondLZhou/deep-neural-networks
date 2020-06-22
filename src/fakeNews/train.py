from sklearn.linear_model import PassiveAggressiveClassifier

def train(tfidf_train, y_train, tfidf_test):
    pac = PassiveAggressiveClassifier(max_iter=50)
    pac.fit(tfidf_train, y_train)

    y_pred = pac.predict(tfidf_test)

    return y_pred
