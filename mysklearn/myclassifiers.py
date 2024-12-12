"""
Name: Abby Hidalgo and Zobe Murray
Course: CPSC 322
Date: 12/11/24
Description: This program contains different classifiers and the fit and predict functionalities of each classifier.

"""

import operator
import mysklearn.myutils as myutils
import mysklearn.myevaluation as myeval
import copy
import numpy as np

class MyRandomForestClassifier:
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
    def __init__(self):
        "Initializer for RandomForestClassifier"
        self.trees = []
        self.headers = []

    def fit(self,X_train,y_train, N, F, M):
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
        best_trees = []
        for _ in range(N):
            X_training, X_val, y_training, y_val = myeval.bootstrap_sample(X_train, y_train)
            tree_class = MyDecisionTreeClassifier()
            tree_class.fit(X_training, y_training, F)
            tree_pred = tree_class.predict(X_val)
            acc = myeval.accuracy_score(y_val, tree_pred)
            if len(best_trees) < M:
                best_trees.append((acc, tree_class.tree, tree_class.header))
                best_trees.sort(reverse=True, key=lambda x: x[0])
            else:
                if acc > best_trees[-1][0]:
                    best_trees[-1] = (acc, tree_class.tree, tree_class.header)
                    best_trees.sort(reverse=True, key=lambda x: x[0])
        self.trees = [tree[1] for tree in best_trees]
        self.headers = [tree[2] for tree in best_trees]
    def predict(self, X_test):
        """Makes a prediction based on the trees fit to the random forest classifier which uses majority voting
        to make the best predict 
        
        Notes:
          Use majority voting to predict classes using M decision trees over test set  
        """    
        predictions = []
        y_preds = []
        for i, tree in enumerate(self.trees):
            tree_class = MyDecisionTreeClassifier()
            tree_class.tree = tree
            tree_class.header = self.headers[i]
            y_pred = tree_class.predict(X_test)
            predictions.append(y_pred)
        for i in range(len(X_test)):
            temp_pred = []
            for prediction in predictions:
                temp_pred.append(prediction[i])
            unique_labels, counts = myutils.get_frequency(temp_pred)
            majority_label = unique_labels[np.argmax(counts)]
            y_preds.append(majority_label)
        return y_preds
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
        self.att_domains = None
        self.prev = None
        self.header = None
        self.maj = None

    def fit(self, X_train, y_train, F):
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
        #initializing X_train and y_train
        self.X_train = X_train
        self.y_train = y_train
        self.maj = self.get_majority_class(X_train)
        #concatinating the training data
        train = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
        #extracting the headers and attribute domains
        num_attributes = len(X_train[0])
        headers = []
        attribute_domains = {}
        for i in range(num_attributes):
            headers.append(f"att{i}")
            attribute_domains[f"att{i}"] = set()
        for row in train:
            for i in range(num_attributes):
                attribute_domains[f"att{i}"].add(row[i])
        self.header = headers
        available_attributes = headers.copy()
        self.att_domains = attribute_domains
        #calling recursive function to build the tree
        tree = self.tdidt(train, available_attributes.copy(), F)
        self.tree = tree

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        #getting predictions for the X_test instances
        predictions = []
        for inst in X_test:
            #calling recursive prediction function
            pred = self.tdidt_predict(inst)
            predictions.append(pred)
        return predictions
    def tdidt(self, current_instances, available_attributes, F):
        """Recursively creates a decision tree.

        Args:
            current_instances(list of list of obj): Instances available for partitioning
            availabel_attributes(list of obj): attributes avaiable to split on

        Returns:
            tree(list of lists): The tree in the form of a nested list
        """
        #getting attribute to split on based on entropy
        np.random.seed(0)
        copy_avaliable = copy.deepcopy(available_attributes)
        if len(copy_avaliable) > F:
            f_attributes = np.random.choice(copy_avaliable, F, False)
        else:
            f_attributes = copy_avaliable
        # select an attribute to split on
        split_attribute = self.select_attribute(current_instances, f_attributes)
        #removing the attribute that is being split on
        available_attributes = [att for att in available_attributes if att != split_attribute]
        tree = ["Attribute", split_attribute]
        self.prev = len(current_instances)
        #partitioning the instances based on their value for the split attribute
        partitions = self.partition_instances(current_instances, split_attribute)
        part_not_empty = True
        value_subtrees = []
        for att_val, att_partition in partitions.items():
            #if a partition is empty, create a leaf node instead of splitting on the split attribute
            #base case 3
            if len(att_partition) == 0:
                part_not_empty = False
                break
            value_subtree = ["Value", att_val]
            #base case 1
            if len(att_partition) > 0 and self.all_same_class(att_partition):
                leaf_class = self.get_majority_class(att_partition)
                value_subtree.append(["Leaf", leaf_class, len(att_partition), len(current_instances)])
            #base case 2
            elif len(att_partition) > 0 and len(available_attributes) == 0:
                majority_class = self.get_majority_class(att_partition)
                value_subtree.append(["Leaf", majority_class, len(att_partition), len(current_instances)])
            else:
                #recursively adding to tree and then appending it to the treee
                subtree = self.tdidt(att_partition, available_attributes, F)
                value_subtree.append(subtree)
            value_subtrees.append(value_subtree)
        if part_not_empty:
            #add the split to tree
            tree.extend(value_subtrees)
        else:
            #add leaf to tree
            majority_class = self.get_majority_class(current_instances)
            tree = ["Leaf", majority_class, len(current_instances), self.prev]
        return tree

    def select_attribute(self, current_instances, available_attributes):
        """Selects an attribute to split on.

        Args:
            current_instances(list of list of obj): Instances available for partitioning
            availabel_attributes(list of obj): attributes avaiable to split on

        Returns:
            attribute_to_split(object): attribute that will be used to split the data
        """
        #getting the unique classes in the current_instances
        unique_classes = {inst[-1] for inst in current_instances}
        best_entropy = float('inf')
        attrbute_to_split = None
        #getting the best attribute (the attribute with the lowest entropy)
        for att in available_attributes:
            att_index = int(att[3:])
            total = len(current_instances)
            weighted_entropy = 0
            #going through all of the values in the attribute
            for domain in self.att_domains[att]:
                subset = [inst for inst in current_instances if inst[att_index] == domain]
                subset_size = len(subset)
                if subset_size == 0:
                    continue
                class_counts = {class_name: 0 for class_name in unique_classes}
                for inst in subset:
                    class_counts[inst[-1]] += 1
                domain_entropy = self.entropy(class_counts, subset_size)
                weighted_entropy += (subset_size / total) * domain_entropy
            #if the new weighted entropy is smaller than the current lowest, the attribute becomes
            #the one to split on
            if weighted_entropy < best_entropy:
                best_entropy = weighted_entropy
                attrbute_to_split = att
        return attrbute_to_split
    def partition_instances(self, current_instances, split_att):
        """Partitions the instances based on the split attribute.

        Args:
            current_instances(list of list of obj): Instances available for partitioning
            split_att(str): attribute to split on

        Returns:
            partitions(list of list): the instances split up based on their attribute value
        """
        att_index = int(split_att[3:])
        att_domain = self.att_domains[split_att]
        partitions = {}
        #splitting instances based on the value for the attribute
        for att_val in sorted(att_domain, key=lambda x: str(x)):
            partitions[att_val] = []
            for instance in current_instances:
                if str(instance[att_index]) == str(att_val):
                    partitions[att_val].append(instance)
        return partitions

    def all_same_class(self, current_instances):
        """Tests if all the instances have the same class.

        Args:
            current_instances(list of list of obj): Instances that are in partition

        Returns:
            boolean: True or False whether or not the instances all have the same class
        """
        first_class = current_instances[0][-1]
        for instance in current_instances:
            if instance[-1] != first_class:
                return False
        return True

    def get_majority_class(self, instances):
        """Finds the class the majority of the instances have, unless there is a tie in which case the class
        that comes first alphabetically becomes the majority class.

        Args:
            current_instances(list of list of obj): Instances that may have the same class

        Returns:
            majority class(object): class that is true for the majority of instances
        """
        class_counts = {}
        #getting how many times each class is present in instances
        for inst in instances:
            inst_class = inst[-1]
            class_counts[inst_class] = class_counts.get(inst_class, 0) + 1
        #finding the class with the highest count of occurances
        max_count = max(class_counts.values())
        #if tied return the class that comes first alphabetically
        tied_classes = [inst_class for inst_class, count in class_counts.items() if count == max_count]
        return min(tied_classes)
    def entropy(self, class_counts, total):
        """Calculates entropy.

        Args:
            class_counts(set): the count of occurances for each class
            availabel_attributes(list of obj): attributes avaiable to split on

        Returns:
            total(int): size of the current partition
        """
        entropy_val = 0
        for count in class_counts.values():
            if count > 0:
                probability = count / total
                entropy_val -= probability * np.log2(probability)
        return entropy_val
    def special_case(self, partitions):
        """Handles case of not enough instances in a partition by calculating denomenator for
        leaf.

        Args:
            partitions(list of lists): the partitions for the current split attribute

        Returns:
            instances(int): Number of instances at the split attribute
        """
        instances = 0
        for _, partition in partitions.items():
            instances += len(partition)
        return instances
    def tdidt_predict(self, instance, trees=None):
        """Recursively traverses the tree until hitting a leaf node.

        Args:
            instances(list): an unseen instance to predict on
            trees(list of lists): current subtree of the decision tree

        Returns:
            pred(string): class predicted
        """
        if trees is None:
            trees = self.tree.copy()
        #test base case of finding a leaf node
        info_type = trees[0]
        if info_type == "Leaf":
            return trees[1]
        #keep recursively traversing
        att_index = self.header.index(trees[1])
        for i in range(2, len(trees)):
            value_list = trees[i]
            if value_list[1] == instance[att_index]:
                return self.tdidt_predict(instance, value_list[2])
        return self.maj

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
        #calling recursive function to create the rules
        self.tdidt_rules(self.tree.copy(), [], class_name, attribute_names)

    def tdidt_rules(self, tree, conditions, class_name, attribute_names):
        """Recursively creates a tree's decision rules.

        Args:
            tree(list of lists): tree to traverse
            conditions(list of obj): current conditions made for rule
            class_name(str): what the user wants the class (RHS) to be called
            attribute_names(list): what the user wants the attribute names to be on the LHS
        """
        #base case where we have reached a leaf and fully create and print the rule
        node_type = tree[0]
        if node_type == "Leaf":
            rule = " AND ".join(conditions)
            label = tree[1]
            print(f"IF {rule} THEN {class_name} = {label}")
            return
        #getting the conditions for each rule
        if node_type == "Attribute":
            attribute = tree[1]
            for subtree in tree[2:]:
                value = subtree[1]
                condition = f"{attribute_names[int(attribute[3:])]} = {value}"
                self.tdidt_rules(subtree[2], conditions + [condition], class_name, attribute_names)
'''
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

    def fit(self, X_train, y_train, F):
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
        self.tree = self.tdidt(instances, header, F)

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
   
    def tdidt(self,current_instances, available_attributes, F):
        """Recursive tdidt algorithm that creates the decision tree

        Args:
            current_instances(list of list): rows of X_train that were passed in
            available_attributes(list): available attributes in the data to split on
        Returns:
            best_att(string): attribute with the lowest attribute
        """
        np.random.seed(0)
        copy_avaliable = copy.deepcopy(available_attributes)
        if len(copy_avaliable) > F:
            f_attributes = np.random.choice(copy_avaliable, F, False)
        else:
            f_attributes = copy_avaliable
        # select an attribute to split on
        split_attribute = self.select_attribute(current_instances, f_attributes)
        copy_avaliable.remove(split_attribute) # can't split on this attribute again
        # in this subtree
        tree = ["Attribute", split_attribute]
        # group data by attribute domains (creates pairwise disjoint partitions)
        partitions = self.partition_instances(current_instances, split_attribute)
        # for each partition, repeat unless one of the following occurs (base case)
        for att_value in sorted(partitions.keys(), key=lambda x: str(x)): # process in alphabetical order
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
                subtree = self.tdidt(att_partition, copy_avaliable.copy(), F)
                value_subtree.append(subtree)
            tree.append(value_subtree)
        return tree
'''    
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
