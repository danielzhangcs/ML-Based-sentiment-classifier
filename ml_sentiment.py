import sys

import numpy as np
from utils import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def get_tuned_lr(train, dev, features, output_file_path='./lr.png'):
    train_vecs = features.fit_transform(train.reviews)
    dev_vecs = features.transform(dev.reviews)
    train_accuracy_list = list()
    dev_accuracy_list = list()
    cs = np.arange(0.01, 0.1, 0.005)  # You will change this range, or may want to use np.logspace instead of np.arrange
    for c in cs:
        model = LogisticRegression(C=c)
        model.fit(train_vecs, train.labels)
        train_preds = model.predict(train_vecs)
        dev_preds = model.predict(dev_vecs)
        (train_score, dev_score) = (accuracy_score(train.labels, train_preds), accuracy_score(dev.labels, dev_preds))
        print("Train Accuracy:", train_score, ", Dev Accuracy:", dev_score)
        train_accuracy_list.append(train_score)
        dev_accuracy_list.append(dev_score)
    plot(cs, train_accuracy_list, dev_accuracy_list, output_file_path)
    best_model = LogisticRegression(C=cs[np.argmax(dev_accuracy_list)])
    return get_trained_classifier(train, best_model, features)


def get_tuned_rf(train, dev, features, output_file_path='./rf.png'):
    train_vecs = features.fit_transform(train.reviews)
    dev_vecs = features.transform(dev.reviews)
    train_accuracy_list = list()
    dev_accuracy_list = list()
    n_estimators = np.arange(120, 200, 10)  # You will change this range, and try different parameters to tune RF model
    for num_estimator in n_estimators:
        model = RandomForestClassifier(n_estimators=num_estimator)
        model.fit(train_vecs, train.labels)
        train_preds = model.predict(train_vecs)
        dev_preds = model.predict(dev_vecs)
        (train_score, dev_score) = (accuracy_score(train.labels, train_preds), accuracy_score(dev.labels, dev_preds))
        print("Train Accuracy:", train_score, ", Dev Accuracy:", dev_score)
        train_accuracy_list.append(train_score)
        dev_accuracy_list.append(dev_score)
    plot(n_estimators, train_accuracy_list, dev_accuracy_list, output_file_path)
    best_model = RandomForestClassifier(n_estimators=n_estimators[np.argmax(dev_accuracy_list)])
    return get_trained_classifier(train, best_model, features)

if __name__ == "__main__":
    filedir = sys.argv[1] if len(sys.argv) > 1 else 'data'
    print("Reading data")
    train_data, dev_data = get_training_and_dev_data(filedir)
    # Some example code to get a trained classifier
    print("Training model")
    lr_with_default = get_trained_classifier(train_data, LogisticRegression(), CountVectorizer())
    #rf_with_default = get_trained_classifier(train_data, RandomForestClassifier(), CountVectorizer())

    #You can see some of the predictions of the classifier by running the following code
    print(lr_with_default.predict(["This movie like!", "This movie is great!"]))
    #print(rf_with_default.predict(["This movie sucks!", "This movie is great!"]))
 
    # You can then experiment with tuning the classifiers
    # Experiment with the parameters in the get_tuned_lr and get_tuned_rf methods
    # Look at the files lr.png and rf.png that are saved after running each of these functions below
    #print("Tuning model")
    #tuned_lr = get_tuned_lr(train_data, dev_data, CountVectorizer())
    #tuned_rf = get_tuned_rf(train_data, dev_data, CountVectorizer())

    # After playing with the parameters and finding a good classifier, you can save
    # This will save the classifier to a pickle object which you can then load later from when doing your error analysis
    # As well as this will run the classifier on the test set which you can then upload to kaggle
    #print("Saving model and predictions")
    #save(tuned_lr, filedir, 'lr_default')
    #save(tuned_rf, filedir, 'rf_default')

    # Then experiment with different features by modifiying custom_features.py and test your accuracy by running:
    # (Again, you can look at the lr.png  and rf.png that are saved after running each of these functions)
    print("Tuning model")
    #tuned_lr = get_tuned_lr(train_data, dev_data, get_custom_features(filedir))
    tuned_rf = get_tuned_rf(train_data, dev_data, get_custom_features(filedir))

    # As before, after playing with different features and finding a good classifier, you can save
    print("Saving model and predictions")
    #save(tuned_lr, filedir, 'lr_custom')
    save(tuned_rf, filedir, 'rf_custom')






























