import pandas as pd

import split
import train
import assess

df = pd.read_csv('../../data/news.csv')
labels = df.label

tfidf_train, y_train, tfidf_test, y_test = split.split(df['text'], labels)

y_pred = train.train(tfidf_train, y_train, tfidf_test)

assess.assess(y_test, y_pred)
