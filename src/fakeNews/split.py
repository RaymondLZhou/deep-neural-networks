from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def split(text, labels):
    x_train, x_test, y_train, y_test = train_test_split(text, labels, test_size=0.2, random_state=7)

    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

    tfidf_train = tfidf_vectorizer.fit_transform(x_train) 
    tfidf_test = tfidf_vectorizer.transform(x_test)

    return tfidf_train, y_train, tfidf_test, y_test
