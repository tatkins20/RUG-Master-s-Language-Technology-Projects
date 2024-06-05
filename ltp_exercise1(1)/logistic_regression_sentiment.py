__author__ = "Barbara Plank. Adapted by Antonio Toral, Trevor Atkins"

"""
Exercise: sentiment classification with logistic regression

1) Examine the code/data.
   What is the distribution of labels in the data (how many positive/negative)?
   What vectorizer is used/how is the text represented?
2) What is the simplest baseline the system can be compared to? Implement the baseline.
3) Add code to train and evaluate the classifier. What accuracy do you get? What is weird?
4) Add code that shows the wrongly predicted instances.

Criteria: Q1a, Q1b, Q2, Q3 and Q4 are worth 2 points each.
"""
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import numpy as np
import random

def load_sentiment_sentences_and_labels():
    """
    loads the movie review data
    """
    positive_sentences = [l.strip() for l in open("rt-polaritydata/rt-polarity.pos").readlines()]
    negative_sentences = [l.strip() for l in open("rt-polaritydata/rt-polarity.neg").readlines()]

    positive_labels = [1 for sentence in positive_sentences]
    negative_labels = [0 for sentence in negative_sentences]

    sentences = np.concatenate([positive_sentences,negative_sentences], axis=0)
    labels = np.concatenate([positive_labels,negative_labels],axis=0)

    ## make sure we have a label for every data instance
    assert(len(sentences)==len(labels))
    data = list(zip(sentences,labels))

    return data

def train_dev_test_split(data):
    """
    split data into train (60%), dev (20%) and test (20%)
    """
    random.seed(113) #seed
    random.shuffle(data)


    # split data
    train_end = int(0.6 * len(data))
    dev_end = int(0.8 * len(data))

    sentences = [sentence for sentence, label in data]
    labels = [label for sentence, label in data]
    X_train, X_dev, X_test = sentences[:train_end], sentences[train_end:dev_end], sentences[dev_end:]
    y_train, y_dev, y_test = labels[:train_end], labels[train_end:dev_end], labels[dev_end:]
    assert (len(X_train) == len(y_train))
    assert (len(X_dev) == len(y_dev))
    assert (len(X_test) == len(y_test))

    return X_train, y_train, X_dev, y_dev, X_test, y_test


## read input data
print("load data..")
data = load_sentiment_sentences_and_labels()
X_train, y_train, X_dev, y_dev, X_test, y_test = train_dev_test_split(data)

print("#train instances: {} #dev: {} #test: {}".format(len(X_train),len(X_dev),len(X_test)))

## Q1a. Distribution of labels in the data
labels = [label for sentence, label in data]
poslbl_cnt = 0
neglbl_cnt = 0
for label in labels:
    if label == 1:
        poslbl_cnt += 1
    elif label == 0:
        neglbl_cnt +=1
print("positive label distribution: " + str(poslbl_cnt))
print("negative label distribution: " + str(neglbl_cnt))
## end of your code for Q1a


## Q2. Add a simple baseline -- your code here

'''
The simplest baseline that the system can be compared to is the
Zero Rule Algorithm, where it predicts the class value most common
in the training dataset.

def mean(listOfValues):
    return(round(sum(listOfValues)/(len(listOfValues))))

def zeroRR(train, test):
    targetValues = [element for element in train]
    meanofTargetValues = mean(targetValues)
    predictions = [meanofTargetValues for i in range(len(test))]
    return predictions
'''
def zeroRR(train, test):
	targetValues = [element for element in train]
	prediction = max(set(targetValues), key=targetValues.count)
	predicted = [prediction for i in range(len(test))]
	return predicted

baseline_test = zeroRR(y_train, y_test)
baseline_dev = zeroRR(y_train, y_dev)
#print("test: " + str(len(y_test)))
#print("dev: " + str(len(y_dev)))
#print(baseline_test)
#print(baseline_dev)
## end your code for Q2


## Q1b. What does this vectorizer do?
'''
CountVectorizer is a vectorizer that converts a collection of text
documents to a matrix of token counts (sparse representation)

if no a-priori dictionary and analyzer - number of features = vocab size
'''
print("vectorize data..")
vectorizer = CountVectorizer()

classifier = Pipeline( [('vec', vectorizer),
                        ('clf', LogisticRegression(max_iter = 1000))] )

print("train model..")


## Q3. Train and evaluate the classifier -- your code here
classifier.fit(X_train, y_train)
y_predicted_test = classifier.predict(X_test)
y_predicted_dev = classifier.predict(X_dev)
#print(len(y_dev))
#print(len(baseline_dev))
#print(len(y_test))
#print(len(baseline_test))
## end your code here for Q3


###

accuracy_dev = accuracy_score(y_dev, y_predicted_dev)

print("===== dev set ====")
print("Baseline:   {0:.2f}".format(accuracy_score(y_dev, baseline_dev)*100))
print("Classifier: {0:.2f}".format(accuracy_dev*100))


accuracy_test = accuracy_score(y_test, y_predicted_test)

print("===== test set ====")
print("Baseline:   {0:.2f}".format(accuracy_score(y_test, baseline_test)*100))
print("Classifier: {0:.2f}".format(accuracy_test*100))

'''
The accuracy for the dev set baseline is 49.62 and for the dev set classifier
is 74.67.
The accuracy for the test set baseline is 49.93 and for the test set classifier
is 75.20.
What is weird is that the accuracies for both the dev set and the test set are
very similar - I think it would be more common for there to be a bigger margin of
difference where the dev set performs slightly better than the test.
'''


## Q4. Inspect output - add your code here
y_test_np = np.asarray(y_test)
X_test_np = np.asarray(X_test)
misclassified_test = []
misclassified_test = np.where(y_test_np != y_predicted_test)
misclassified_instances = []
true_instances = []

for index in misclassified_test:
    misclassified_instances.append(y_predicted_test[index])
    true_instances.append(y_test_np[index])


print("misclassified predicted instance one: " + str(misclassified_instances[0][0]) + "\n"
+ "true instance one: " + str(true_instances[0][0]) + "\n")
print("misclassified predicted instance two: " + str(misclassified_instances[0][1]) + "\n"
+ "true instance two: " + str(true_instances[0][1]) + "\n")
print("misclassified predicted instance three: " + str(misclassified_instances[0][2]) + "\n"
+ "true instance three: " + str(true_instances[0][2]) + "\n")
print("misclassified predicted instance four: " + str(misclassified_instances[0][3]) + "\n"
+ "true instance four: " + str(true_instances[0][3]) + "\n")
print("misclassified predicted instance five: " + str(misclassified_instances[0][4]) + "\n"
+ "true instance five: " + str(true_instances[0][4]) + "\n")
print("misclassified predicted instance six: " + str(misclassified_instances[0][5]) + "\n"
+ "true instance six: " + str(true_instances[0][5]) + "\n")
print("misclassified predicted instance seven: " + str(misclassified_instances[0][6]) + "\n"
+ "true instance seven: " + str(true_instances[0][6]) + "\n")
print("misclassified predicted instance eight: " + str(misclassified_instances[0][7]) + "\n"
+ "true instance eight: " + str(true_instances[0][7]) + "\n")
print("misclassified predicted instance nine: " + str(misclassified_instances[0][8]) + "\n"
+ "true instance nine: " + str(true_instances[0][8]) + "\n")
print("misclassified predicted instance ten: " + str(misclassified_instances[0][9]) + "\n"
+ "true instance ten: " + str(true_instances[0][9]) + "\n")

## end
