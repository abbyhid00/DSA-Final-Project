"""
Name: Zobe Murray
Course: CPSC 322
Date: 10/18/2024
Description: This file implements useful functions for myclassifiers and pa6 files

"""
import numpy as np
import tabulate
import myevaluation as myeval
#from mysklearn.myclassifiers import MyKNeighborsClassifier
#from mysklearn.myclassifiers import MyDummyClassifier

def compute_euclidean_distance(v1, v2):
    """Computes the euclidean distance for two values.

    Args:
        v1, v2(float): values we want to compute the euclidean distance between

    Returns: 
        distance(float): calculated distance
    """
    #if not all(isinstance(i, (int, float)) for i in v1) or not all(isinstance(i, (int, float)) for i in v2):
        #for i in range(len(v1)):
            #if v1[i] == v[]
    return  np.sqrt(sum([(v1[i] - v2[i]) ** 2 for i in range(len(v1))]))

def get_frequency(y_train):
    """Computes the frequency for each unique value of a list.

    Args:
        y_train(list): list of values we want to find the frequency in

    Returns: 
        unique_vals(list): list of the unique values in the training data
        counts(list): the number of times each unique value shows up
    """
    unique_vals = sorted(list(set(y_train)))
    counts = []
    for val in unique_vals:
        counts.append(y_train.count(val))
    return unique_vals, counts


def randomize_in_place(alist, parallel_list=None):
    """Gina Sprint's randomize in place function.

    Args:
        alist(list): list of values we want shuffled
        parallel_list(list): list we want shuffled in parallel

    Returns:
        alist, parallel_list(lists): the shuffled lists
    """
    for i in range(len(alist)):
        # generate a random index to swap this value at i with
        rand_index = np.random.randint(0, len(alist)) # rand int in [0, len(alist))
        # do the swap
        alist[i], alist[rand_index] = alist[rand_index], alist[i]
        if parallel_list is not None:
            parallel_list[i], parallel_list[rand_index] = parallel_list[rand_index], parallel_list[i]

def group_by(y):
    """Groups instances' indices into groups based on their class label.

    Args:
        X(list of lists): list of instances we want grouped
        y(list of obj): the class labels of X's instances

    Returns: 
        grouped(list of lists): list of indices separated based on their label
    """
    #Get all of the unique labels
    unique_values, counts = get_frequency(y)
    grouped = []
    #For each label, if the y label is the same as
    #The current label add its index to the group
    for val in unique_values:
        label = []
        for i, y_val in enumerate(y):
            if y_val == val:
                label.append(i)
        grouped.append(label)
    return grouped


def get_train_test(X, y, fold):
    """Separates the training and testing data from the tuple returned in 
    cross validation.

    Args:
        X(list of lists): list of instances we want to split
        y(list): class labels for the instances in X
        fold(tuple): tuple that holds the training and testing data for fold

    Returns: 
        X_train(list of lists): training instances
        X_test(list of lists): testing instances
        y_train(list): class labels for training data
        y_test(list): class labels for testing data
    """
    #split tuple into training and testing
    train = fold[0]
    test = fold[1]
    X_test = []
    X_train = []
    y_test = []
    y_train = []
    #Create the test and training data
    for inst in train:
        X_train.append(X[inst])
        y_train.append(y[inst])
    for inst in test:
        X_test.append(X[inst])
        y_test.append(y[inst])
    return X_train, X_test, y_train, y_test

def calc_av_acc(accuracy):
    """Calculates the average accuracy of a list.

    Args:
        accuracy(list): list of accuracy scores

    Returns: 
        sum_of_acc / len(accuracy)(float): average accuracy of list
    """
    sum_of_acc = 0
    for num in accuracy:
        sum_of_acc += num
    return sum_of_acc / len(accuracy)


def print_results(title, method, knn_acc, knn_err, dum_acc, dum_err):
    """Prints out the results of the training and testing methods and classifiers.

    Args:
        title(string): title
        method(string): methods being used to produce data
        knn_acc(float): accuracy from knn classifier
        knn_err(float): error rate from knn classifier
        dum_acc(float): accuracy from dummy classifier
        dum_err(float): error rate from the dummy classifier
    """
    print("================================================")
    print(title)
    print("================================================")
    print(method)
    print("k Nearest Neighbors Classifier: accuracy =", round(knn_acc, 2), "error rate =", round(knn_err, 2))
    print("Dummy Classifier: accuracy =", round(dum_acc, 2), "error rate =", round(dum_err, 2))

def print_confusion(title, method, y_pred, y_test):
    """Prints confusion matrix.

    Args:
        title(string): title
        method(string): methods used to produce data
        y_pred(list): list of predictions
        y_test(list): list of actual class labels
    """
    #labels for confusion matrix
    labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    #getting the confusion matrix for y_pred and y_test
    matrix = myeval.confusion_matrix(y_test, y_pred, labels)
    matrix_extra = []
    #calculating total and recognition percentage for extra columns
    for i, row in enumerate(matrix):
        total = sum(row)
        recognition = (row[i] / total) * 100 if total > 0 else 0
        matrix_extra.append(row + [total, f"{recognition:.2f}%"])
    #intializing headers for matrix
    headers = ["MPG Ranking"] + labels + ["Total", "Recognition (%)"]
    matrix_labels = [[str(label)] + row for label, row in zip(labels, matrix_extra)]
    #printing matrix
    print("================================================")
    print(title)
    print("================================================")
    print(method)
    print(tabulate.tabulate(matrix_labels, headers=headers))

def compute_category_distance(v1, v2):
    """Computes distance for categorical attributes.

    v1, v2(float): values we want to compute the euclidean distance between

    Returns: 
        distance(float): calculated distance
    """
    dist = 0
    for _ in range(len(v1)):
        dist += 0 if v1 == v2 else 1
    return dist ** 0.5

def print_metrics(y_pred, y_true):
    """Prints metrics from classifier.

    y_pred(list): list of predictions
    y_test(list): list of actual class labels
    """
    #getting labels
    unique_vals, counts = get_frequency(y_true)
    #getting metrics
    acc = myeval.accuracy_score(y_true, y_pred)
    err = 1 - acc
    prec = myeval.binary_precision_score(y_true, y_pred)
    rec = myeval.binary_recall_score(y_true, y_pred)
    f1 = myeval.binary_f1_score(y_true, y_pred)
    #printing metrics
    print("Accuracy:", round(acc, 2))
    print("Error Rate:", round(err, 2))
    print("Precision:", round(prec, 2))
    print("Recall:", round(rec, 2))
    print("F1 Score:", round(f1, 2))
    print("Confusion Matrix:")
    matrix = myeval.confusion_matrix(y_true, y_pred, unique_vals)
    matrix_extra = []
    #calculating total and recognition percentage for extra columns
    for i, row in enumerate(matrix):
        total = sum(row)
        recognition = (row[i] / total) * 100 if total > 0 else 0
        matrix_extra.append(row + [total, f"{recognition:.2f}%"])
    #intializing headers for matrix
    headers = unique_vals + ["Total", "Recognition (%)"]
    matrix_labels = [[str(label)] + row for label, row in zip(unique_vals, matrix_extra)]
    #printing matrix
    print(tabulate.tabulate(matrix_labels, headers=headers))
    