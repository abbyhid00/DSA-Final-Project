
import numpy as np # use numpy's random number generation
import analysis_code.myutils as myutils
import tabulate
# TODO: copy your myevaluation.py solution from PA5 here

def binary_precision_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the precision (for binary classification). The precision is the ratio tp / (tp + fp)
        where tp is the number of true positives and fp the number of false positives.
        The precision is intuitively the ability of the classifier not to label as
        positive a sample that is negative. The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        precision(float): Precision of the positive class

    Notes:
        Loosely based on sklearn's precision_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    """
    TP = 0
    FP = 0
    #getting labels and positive label if not given one
    if labels is None:
        unique_val, counts = myutils.get_frequency(y_true)
        labels = unique_val
    if pos_label is None:
        pos_label = labels[0]
    #getting TP and FP
    for i in range(len(y_pred)):
        if y_pred[i] == y_true[i] and y_true[i] == pos_label:
            TP += 1
        if y_pred[i] == pos_label and y_true[i] != pos_label:
            FP += 1
    #calculating the precision
    if (TP + FP) > 0:
        precision = TP / (TP + FP)
    else:
        precision = 0.0
    return round(precision, 2)

def binary_recall_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the recall (for binary classification). The recall is the ratio tp / (tp + fn) where tp is
        the number of true positives and fn the number of false negatives.
        The recall is intuitively the ability of the classifier to find all the positive samples.
        The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        recall(float): Recall of the positive class

    Notes:
        Loosely based on sklearn's recall_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    """
    TP = 0
    FN = 0
    #getting labels and positive label if not given
    if labels is None:
        unique_val, counts = myutils.get_frequency(y_true)
        labels = unique_val
    if pos_label is None:
        pos_label = labels[0]
    #finding the TP and FN
    for i in range(len(y_pred)):
        if y_pred[i] == y_true[i] and y_true[i] == pos_label:
            TP += 1
        if y_pred[i] != pos_label and y_true[i] == pos_label:
            FN += 1
    #calculating recall
    if (TP + FN) > 0:
        recall = TP / (TP + FN)
    else:
        recall = 0.0
    return round(recall, 2)

def binary_f1_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the F1 score (for binary classification), also known as balanced F-score or F-measure.
        The F1 score can be interpreted as a harmonic mean of the precision and recall,
        where an F1 score reaches its best value at 1 and worst score at 0.
        The relative contribution of precision and recall to the F1 score are equal.
        The formula for the F1 score is: F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        f1(float): F1 score of the positive class

    Notes:
        Loosely based on sklearn's f1_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    """
    #getting labels and positive label if not given
    if labels is None:
        unique_val, counts = myutils.get_frequency(y_true)
        labels = unique_val
    if pos_label is None:
        pos_label = labels[0]
    #getting recall and precision
    recall = binary_recall_score(y_true, y_pred, labels, pos_label)
    precision = binary_precision_score(y_true, y_pred, labels, pos_label)
    #calculating f1 score
    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
    return round(f1, 2)

def stratified_kfold_split(X, y, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into stratified cross validation folds.

    Args:
        X(list of list of obj): The list of instances (samples).
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X).
            The shape of y is n_samples
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds

    Returns:
        folds(list of 2-item tuples): The list of folds where each fold is defined as a 2-item tuple
            The first item in the tuple is the list of training set indices for the fold
            The second item in the tuple is the list of testing set indices for the fold

    Notes:
        Loosely based on sklearn's StratifiedKFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    #setting seed
    if random_state is None:
        np.random.seed(0)
    else:
        np.random.seed(random_state)
    #shuffling if shuffle is true
    if shuffle:
        myutils.randomize_in_place(X, y)
    #grouping the instances by class
    grouped = myutils.group_by(y)
    #creating the folds
    folds = [[] for _ in range(n_splits)]
    #calculating how many instances will go into each fold
    for label in grouped:
        fold_sizes = [len(label) // n_splits] * n_splits
        for i in range(len(label) % n_splits):
            fold_sizes[i] += 1
        current = 0
        for fold, fold_size in enumerate(fold_sizes):
            folds[fold].extend(label[current:current + fold_size])
            current += fold_size
    kfolds = []
    #adding test and train data to each fold
    for fold in range(n_splits):
        X_test = folds[fold]
        X_train = [index for i, f in enumerate(folds) if i != fold for index in f]
        kfolds.append((X_train, X_test))

    return kfolds
def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry
            indicates the number of samples with true label being i-th class
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    #Creating matrix that has entries for each label
    matrix = [[0 for _ in range(len(labels))] for _ in range(len(labels))]
    label_dic = {label: i for i, label in enumerate(labels)}
    #inputting the values for each matrix index
    for true, pred in zip(y_true, y_pred):
        true_index = label_dic[true]
        pred_index = label_dic[pred]
        matrix[true_index][pred_index] +=1


    return matrix

def accuracy_score(y_true, y_pred, normalize=True):
    """Compute the classification prediction accuracy score.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        normalize(bool): If False, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.

    Returns:
        score(float): If normalize == True, return the fraction of correctly classified samples (float),
            else returns the number of correctly classified samples (int).

    Notes:
        Loosely based on sklearn's accuracy_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    """
    correct = 0
    #if predicted and true value are the same add to
    #number of correct
    for i in range(len(y_pred)):
        if y_true[i] == y_pred[i]:
            correct += 1
    #if normalize is true return the accuracy
    if normalize:
        return correct / len(y_pred)
    #if not true return the number of correct predictions
    else:
        return correct
    
def bootstrap_sample(X, y=None, n_samples=None, random_state=None):
    """Split dataset into bootstrapped training set and out of bag test set.

    Args:
        X(list of list of obj): The list of samples
        y(list of obj): The target y values (parallel to X)
            Default is None (in this case, the calling code only wants to sample X)
        n_samples(int): Number of samples to generate. If left to None (default) this is automatically
            set to the first dimension of X.
        random_state(int): integer used for seeding a random number generator for reproducible results

    Returns:
        X_sample(list of list of obj): The list of samples
        X_out_of_bag(list of list of obj): The list of "out of bag" samples (e.g. left-over samples)
        y_sample(list of obj): The list of target y values sampled (parallel to X_sample)
            None if y is None
        y_out_of_bag(list of obj): The list of target y values "out of bag" (parallel to X_out_of_bag)
            None if y is None
    Notes:
        Loosely based on sklearn's resample():
            https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html
        Sample indexes of X with replacement, then build X_sample and X_out_of_bag
            as lists of instances using sampled indexes (use same indexes to build
            y_sample and y_out_of_bag)
    """
    if random_state is not None:
        np.random.seed(random_state)
    if n_samples is None:
        n_samples = len(X)
    chosen_indices = np.random.randint(0, len(X), size=n_samples)
    chosen_indices_set = set(chosen_indices)
    all_indices_set = set(range(len(X)))
    out_bag_indices = all_indices_set - chosen_indices_set #finds # of instances where they aren't chosen
    X_sample = [X[i] for i in chosen_indices]
    y_sample = [y[i] for i in chosen_indices]
    X_out_of_bag = [X[i] for i in out_bag_indices]
    y_out_of_bag = [y[i] for i in out_bag_indices]

    return X_sample, X_out_of_bag, y_sample, y_out_of_bag

def classification_report(y_true, y_pred, out_dict, labels=None):
    """Build a text report and a dictionary showing the main classification metrics.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        output_dict(bool): If True, return output as dict instead of a str

    Returns:
        report(str or dict): Text summary of the precision, recall, F1 score for each class.
            Dictionary returned if output_dict is True. Dictionary has the following structure:
                {'label 1': {'precision':0.5,
                            'recall':1.0,
                            'f1-score':0.67,
                            'support':1},
                'label 2': { ... },
                ...
                }
            The reported averages include macro average (averaging the unweighted mean per label) and
            weighted average (averaging the support-weighted mean per label).
            Micro average (averaging the total true positives, false negatives and false positives)
            multi-class with a subset of classes, because it corresponds to accuracy otherwise
            and would be the same for all metrics. 

    Notes:
        Loosely based on sklearn's classification_report():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
    """
    #getting labels
    if labels == None:
        labels = sorted(set(y_true))
    #initializing metrics for averages
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_support = 0
    report_dict = {}
    #creating the dictionary for the classificating report
    for label in labels:
        label_str = str(label)
        report_dict[label_str] = {}
        report_dict[label_str]['precision'] = round(binary_precision_score(y_true, y_pred, labels, label), 2)
        report_dict[label_str]['recall'] = round(binary_recall_score(y_true, y_pred, labels, label), 2)
        report_dict[label_str]['f1-score'] = round(binary_f1_score(y_true, y_pred, labels, label), 2)
        report_dict[label_str]['support'] = y_true.count(label)
        total_precision += report_dict[label_str]['precision'] * report_dict[label_str]['support']
        total_recall += report_dict[label_str]['recall'] * report_dict[label_str]['support']
        total_f1 += report_dict[label_str]['f1-score'] * report_dict[label_str]['support']
        total_support += report_dict[label_str]['support']
    #adding accuracy to the report
    report_dict['accuracy'] = {
        'f1-score' : round(accuracy_score(y_true, y_pred), 2),
        'support' : total_support
    }
    #adding the macro average
    report_dict['macro avg'] = {
        'precision': round(sum(report_dict[str(label)]['precision'] for label in labels) / len(labels), 2),
        'recall': round(sum(report_dict[str(label)]['recall'] for label in labels) / len(labels), 2),
        'f1-score': round(sum(report_dict[str(label)]['f1-score'] for label in labels) / len(labels), 2),
        'support': total_support
    }
    #adding the weighted average
    report_dict['weighted avg'] = {
        'precision': round(total_precision / total_support, 2),
        'recall': round(total_recall / total_support, 2),
        'f1-score': round(total_f1 / total_support, 2),
        'support': total_support
    }
    #returning dictionary or printing out tabulated table
    if out_dict:
        return report_dict
    else:
        headers = ["", 'precision', 'recall', 'f1-score', 'support']
        table = [
            [label, metrics.get('precision', ''), metrics.get('recall', ''), metrics.get('f1-score', ''), metrics.get('support', '')]
            for label, metrics in report_dict.items()
        ]
        report = tabulate.tabulate(table, headers=headers, tablefmt='grid')
        return report