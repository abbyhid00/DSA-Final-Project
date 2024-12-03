import operator
import numpy as np
import analysis_code.myutils as myutils

class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
            label in the training set.
        posteriors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The posterior probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        You may add additional instance attributes if you would like, just be sure to update this docstring
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.
        """
        self.priors = None
        self.posteriors = None
        self.unique_class = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """
        #getting unique labels and number of instances in the label
        self.unique_class, counts = myutils.get_frequency(y_train)
        samples = len(y_train)
        self.priors = {}
        self.posteriors = {}
        #creating the priors
        for i in range(len(self.unique_class)):
            self.priors[self.unique_class[i]] = [counts[i], samples]
        #initializing the posteriors for each of the labels
        for label in self.unique_class:
            self.posteriors[label] = {}
        #creating the the posteriors for each of the attributs values for
        #each of the labels
        for index in range(len(X_train[0])):
            feature = {label: {} for label in self.unique_class}
            for i in range(len(X_train)):
                instance = X_train[i]
                label = y_train[i]
                feature_val = instance[index]
                if feature_val not in feature[label]:
                    feature[label][feature_val] = 0
                feature[label][feature_val] += 1
            for label in self.unique_class:
                feature_name = f'att{index + 1}'
                if feature_name not in self.posteriors[label]:
                    self.posteriors[label][feature_name] = {}
                total_count = self.priors[label][0]
                for val in feature[label]:
                    count = feature[label][val]
                    self.posteriors[label][feature_name][str(val)] = [count, total_count]

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_pred = []
        #going through each test instance and calculating the prediction
        #based on the label with highest probability
        for inst in X_test:
            highest_prob = 0
            label_to_add = self.unique_class[0]
            #calculating the probability for each label
            for label in self.unique_class:
                prob = 1
                for i, val in enumerate(inst):
                    att = f'att{i + 1}'
                    if att in self.posteriors[label] and str(val) in self.posteriors[label][att]:
                        prob = prob * ((self.posteriors[label][f'att{i + 1}'][str(val)][0]) / (self.posteriors[label][f'att{i + 1}'][str(val)][1]))
                    else:
                        prob = prob * 0
                prob = prob * ((self.priors[label][0]) / (self.priors[label][1]))
                #finding the highest probability
                if prob > highest_prob:
                    highest_prob = prob
                    label_to_add = label
                    #dealing with tie
                elif prob == highest_prob:
                    if ((self.priors[label_to_add][0]) / (self.priors[label_to_add][1])) > (self.priors[label][0]) / (self.priors[label][1]):
                        continue
                    elif ((self.priors[label_to_add][0]) / (self.priors[label_to_add][1])) < (self.priors[label][0]) / (self.priors[label][1]):
                        highest_prob = prob
                        label_to_add = label
                    else:
                        rand = np.random.randint(0, 2)
                        if rand == 0:
                            continue
                        else:
                            highest_prob = prob
                            label_to_add = label
            y_pred.append(label_to_add)
        return y_pred

class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train
        self.y_train = y_train

    def kneighbors(self, X_test):
        """Determines the k closes neighbors of each test instance.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        all_dist = []
        all_neigh = []
        #for each unseen instance compute its euclidean distance
        #from the regression line
        for unseen in X_test:
            row_indexes_dist = []
            for i, row in enumerate(self.X_train):
                dist = myutils.compute_category_distance(row, unseen)
                row_indexes_dist.append((i, dist))
            #find the nearest neigbors (smallest distances)
            row_indexes_dist.sort(key=operator.itemgetter(-1))
            top_k = row_indexes_dist[:self.n_neighbors]
            dist = []
            neigh = []
            #get distance and index of the nearest neighbors
            for instance in top_k:
                dist.append(instance[1])
                neigh.append(instance[0])
            all_dist.append(dist)
            all_neigh.append(neigh)
        return all_dist, all_neigh

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        #getting the knn
        dist, neigh = self.kneighbors(X_test)
        pred = []
        #get the most frequent label for knn and assign it to the unseen instance
        for instance in neigh:
            neigh_labels = [self.y_train[index] for index in instance]
            freq_label = max(set(neigh_labels), key=neigh_labels.count)
            pred.append(freq_label)
        return pred