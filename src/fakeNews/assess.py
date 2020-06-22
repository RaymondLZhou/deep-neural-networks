from sklearn.metrics import accuracy_score, confusion_matrix

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
