import operator
import numpy as np
import analysis_code.myutils as myutils
import copy

class RandomForestClassifier:
    """Represents a random forest classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        trees(nested list): The models of extracted trees from the decision tree classifier.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def ___init__(self):
        "Initializer for RandomForestClassifier"
    def fit(self,X_train,y_train):
        """Fits a random forest classifier to X_train and y_train using bootstrapping to 
        fit the random forest classifier with possible decision trees that could be generated
        
        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            generate a random stratified test set (one third of original data set) and the 
                remaining two thirds of instances form the remainder set
            generate N random decision trees using bootstrapping over the remainder step. Where at
                each node a decision trees should be built randomly by selecting F of the remaining attributes
                to partition on
            Select the M most accurate of the N decision trees using the valiation sets

        """

    def predict(self):
        """Makes a prediction based on the trees fit to the random forest classifier which uses majority voting
        to make the best predict 
        
        Notes:
          Use majority voting to predict classes using M decision trees over test set  
        """    
        
class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.tree = None
        self.header = None
        self.attribute_domains = None

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            On a majority vote tie, choose first attribute value based on attribute domain ordering.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        x_copy = copy.deepcopy(X_train)#deep copy to avoid modifications to self
        header = [f"att{i}" for i in range(len(x_copy[0]))]
        attribute_domains = {}
        for feature_idx in range(len(x_copy[0])):  #go through feature indexes and choose an attribute name
            feature_key = header[feature_idx]
            if feature_key not in attribute_domains:
                attribute_domains[feature_key] = [] #if its not already in the domain than initialize list
            for instance in x_copy: #add instances to dictionary
                feature_val = instance[feature_idx]
                if feature_val not in attribute_domains[feature_key]:
                    attribute_domains[feature_key].append(feature_val)
        self.attribute_domains = attribute_domains
        self.header = header
        instances = [x + [y] for x, y in zip(x_copy, y_train)]
        self.tree = self.tdidt(instances, header)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        tree_copy = self.tree.copy()
        for row in X_test:
            prediction = self.tdidt_predict(tree_copy, row)
            y_predicted.append(prediction)
        return y_predicted

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format
        "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        if attribute_names is None:
            attribute_names = self.header
        self.decision_rule(copy.deepcopy(self.tree,),[],attribute_names,class_name)

    def tdidt_predict(self,tree, instance):
        """Acts as a helper function for predict

        Args:
            tree(deeply nested list): The list of decisions that can be made in the tree.
            instance(list of list): A row from X_train to predict on

        Returns:
            tree[1]: the class label when a leaf is hit or it recursively calls itself
            
        Notes:
        This is adapted from the in class starter code
        """
        # base case: we are at a leaf node and can return the class prediction
        info_type = tree[0] # "Leaf" or "Attribute"
        if info_type == "Leaf":
            return tree[1] # class label
        # if we are here, we are at an Attribute
        # we need to match the instance's value for this attribute
        # to the appropriate subtree
        att_index = self.header.index(tree[1])
        for i in range(2, len(tree)):
            value_list = tree[i]
            # do we have a match with instance for this attribute?
            if type(value_list[1]) != type(instance[att_index]):
                att_string = str(instance[att_index])
            att_string = str(instance[att_index])
            if value_list[1] == att_string:
                return self.tdidt_predict(value_list[2], instance)
      
    def decision_rule(self, tree, value,attribute_names,class_name):
        """Helper function that prints the decision rules and recurses

        Args:
            tree(deeply nested list): The list of decisions that can be made in the tree.
            value(list): list to append the conditions to as it recurses
             attribute_names(list of str or None): A list of attribute names to use in the decision rules
            class_name(str): A string to use for the class name in the decision rules

        Returns:
            prints the decision rule or recurses
        """
        info_type = tree[0]
        if info_type == "Leaf":
            rule = " AND ".join(value)
            return print("IF", rule, "THEN", class_name, " = ", tree[1])
        elif info_type == "Attribute":
            att_val = tree[1]
            for sublist in tree[2:]:
                conditional_val = sublist[1]
                condition = f"{attribute_names[int(att_val[3:])]} = {conditional_val}"
                self.decision_rule(sublist[2], value + [condition],attribute_names, class_name)

    def calc_entropy(self,instances):
        """Calculates the entropy of the partition

        Args:
            instance(list of list): partition that the entropy needs to be found for

        Returns:
            prints the decision rule or recurses
        """
        if not instances:
            return 0
        # get class labels
        labels = [instance[-1] for instance in instances]
        unique_labels, counts = np.unique(labels, return_counts=True)
        probabilities = counts / len(labels)
        return np.sum(-probabilities * np.log2(probabilities))

    def select_attribute(self,instances, attributes):
        """Selects an attribute for the tdidt algorithm to split on

        Args:
            instances(list of list): the rows that need to the paritioned
            attributes(list): potential attributes to split on
        Returns:
            best_att(string): attribute with the lowest attribute
        """
        best_att = None
        low_entropy = float('inf')
        total_inst = len(instances)

        for attribute in attributes:
            weighted_entropy = 0
            partitions = self.partition_instances(instances, attribute)  # partition the data
            for value, partition in partitions.items():
                if len(partition) > 0:
                    entropy = self.calc_entropy(partition)
                    weight = len(partition) / total_inst #calc the weight of partition
                    weighted_entropy += weight * entropy  #add to the weighted sum
            if weighted_entropy < low_entropy:
                low_entropy = weighted_entropy
                best_att = attribute

        return best_att
    
    def partition_instances(self, instances, attribute):
        """Partition the instances by attribute

        Args:
            instances(list of list): the rows that need to the paritioned
            attribute(string): attribute to partition the instances by
        Returns:
            partitions(dictionary): look up table that holds instances for each attribute
            
        Notes: 
        adapted from in class notes
        """
        # this is group by attribute domain (not values of attribute in instances)
        # lets use dictionaries
        att_index = self.header.index(attribute)
        att_domain = self.attribute_domains[attribute]
        partitions = {}
        for att_value in att_domain:
            partitions[att_value] = []
            for instance in instances:
                if instance[att_index] == att_value:
                    partitions[att_value].append(instance)
        return partitions

    def all_same_class(self, instances):
        """checks if instances hav the same class labels or not

        Args:
            instances(list of list): the rows to chek for the same classes
        Returns:
            true or false depending on if matches were found or not
        Notes:
            Adapted from the in class notes
        """
        first_class = instances[0][-1]
        for instance in instances:
            if instance[-1] != first_class:
                return False
        # get here, then all same class labels
        return True

    def majority_vote(self,instances):
        """finds the class label with the highest probability to pick

        Args:
            instances(list of list): the rows that need to be checked for majority votes

        Returns:
            majority_label(string): returns the class label for the attribute that won the majority vote
        """
        labels = [instance[-1] for instance in instances]
        unique_labels, counts = np.unique(labels, return_counts=True)
        majority_label = unique_labels[np.argmax(counts)]
        return majority_label
   
    def tdidt(self,current_instances, available_attributes):
        """Recursive tdidt algorithm that creates the decision tree

        Args:
            current_instances(list of list): rows of X_train that were passed in
            available_attributes(list): available attributes in the data to split on
        Returns:
            best_att(string): attribute with the lowest attribute
        """
        copy_avaliable = copy.deepcopy(available_attributes)
        # select an attribute to split on
        split_attribute = self.select_attribute(current_instances, copy_avaliable)
        copy_avaliable.remove(split_attribute) # can't split on this attribute again
        # in this subtree
        tree = ["Attribute", split_attribute]
        # group data by attribute domains (creates pairwise disjoint partitions)
        partitions = self.partition_instances(current_instances, split_attribute)
        # for each partition, repeat unless one of the following occurs (base case)
        for att_value in sorted(partitions.keys()): # process in alphabetical order
            att_partition = partitions[att_value]
            value_subtree = ["Value", att_value]
            #    CASE 1: all class labels of the partition are the same
            # => make a leaf node
            if len(att_partition) > 0 and self.all_same_class(att_partition):
                class_label = att_partition[0][-1]
                value_subtree.append(["Leaf", class_label,len(att_partition), len(current_instances)])
            #    CASE 2: no more attributes to select (clash)
            # => handle clash w/majority vote leaf node
            elif len(att_partition) > 0 and len(copy_avaliable) == 0:
                class_label = self.majority_vote(current_instances)
                value_subtree.append(["Leaf",class_label,len(att_partition),len(current_instances)])
            #    CASE 3: no more instances to partition (empty partition)
            # => backtrack and replace attribute node with majority vote leaf node
            elif len(att_partition) == 0:
                class_label = self.majority_vote(current_instances)
                value_subtree.append(["Leaf",class_label,0,len(current_instances)])
            else:
                # none of base cases were true, recurse!!
                subtree = self.tdidt(att_partition, copy_avaliable.copy())
                value_subtree.append(subtree)
            tree.append(value_subtree)
        return tree
    
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