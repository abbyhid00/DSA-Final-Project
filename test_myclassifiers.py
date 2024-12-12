# pylint: skip-file
"""test_myevaluation.py
Programmer: Abby Hidalgo
Class: CptS 322-01, Fall 2024
Programming Assignment #7
12/6/24
Description: This is the test file for PA7
"""
import numpy as np
from scipy import stats

from mysklearn.myclassifiers import MyDecisionTreeClassifier,\
    MyNaiveBayesClassifier,\
    MyKNeighborsClassifier,\
    MyRandomForestClassifier
def test_random_forest_classifier_fit():
    header_interview = ["level", "lang", "tweets", "phd", "interviewed_well"]
    X_train_interview = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]
    y_train_interview = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]
    interview_trees = MyRandomForestClassifier()
    interview_trees.fit(X_train_interview, y_train_interview, 20, 2, 7)
    for tree in interview_trees.trees:
        print("New tree:")
        print(tree)
    print(interview_trees.headers)
    assert len(interview_trees.trees) == 7
    assert interview_trees.trees[0] != interview_trees.trees[1]

def test_ranom_forest_classifier_predict():
    """X_test_interview = [["Junior", "Java", "yes", "no"], ["Junior", "Java", "yes", "yes"]]
    interview_trees = MyRandomForestClassifier()
    interview_trees.trees = [[['Attribute', 'att2', ['Value', 'no', ['Attribute', 'att1', ['Value', 'Java', ['Leaf', 'False', 3, 10]], ['Value', 'Python', ['Attribute', 'att0', ['Value', 'Junior', ['Leaf', 'False', 4, 7]], ['Value', 'Mid', ['Leaf', 'True', 2, 7]], ['Value', 'Senior', ['Leaf', 'False', 1, 7]]]], ['Value', 'R', ['Leaf', 'False', 0, 10]]]], ['Value', 'yes', ['Leaf', 'True', 4, 14]]]], 
                             [['Attribute', 'att3', ['Value', 'no', ['Attribute', 'att2', ['Value', 'no', ['Attribute', 'att0', ['Value', 'Junior', ['Leaf', 'True', 3, 7]], ['Value', 'Mid', ['Leaf', 'True', 1, 7]], ['Value', 'Senior', ['Leaf', 'False', 3, 7]]]], ['Value', 'yes', ['Leaf', 'True', 3, 10]]]], ['Value', 'yes', ['Attribute', 'att2', ['Value', 'no', ['Leaf', 'True', 1, 4]], ['Value', 'yes', ['Attribute', 'att0', ['Value', 'Junior', ['Leaf', 'False', 2, 3]], ['Value', 'Mid', ['Leaf', 'True', 1, 3]], ['Value', 'Senior', ['Leaf', 'False', 0, 3]]]]]]]],
                             [['Attribute', 'att3', ['Value', 'no', ['Attribute', 'att2', ['Value', 'no', ['Attribute', 'att0', ['Value', 'Junior', ['Leaf', 'True', 3, 7]], ['Value', 'Mid', ['Leaf', 'True', 1, 7]], ['Value', 'Senior', ['Leaf', 'False', 3, 7]]]], ['Value', 'yes', ['Leaf', 'True', 3, 10]]]], ['Value', 'yes', ['Attribute', 'att2', ['Value', 'no', ['Leaf', 'True', 1, 4]], ['Value', 'yes', ['Attribute', 'att0', ['Value', 'Junior', ['Leaf', 'False', 2, 3]], ['Value', 'Mid', ['Leaf', 'True', 1, 3]], ['Value', 'Senior', ['Leaf', 'False', 0, 3]]]]]]]],
                             [['Attribute', 'att3', ['Value', 'no', ['Attribute', 'att2', ['Value', 'no', ['Attribute', 'att0', ['Value', 'Junior', ['Leaf', 'True', 3, 7]], ['Value', 'Mid', ['Leaf', 'True', 1, 7]], ['Value', 'Senior', ['Leaf', 'False', 3, 7]]]], ['Value', 'yes', ['Leaf', 'True', 3, 10]]]], ['Value', 'yes', ['Attribute', 'att2', ['Value', 'no', ['Leaf', 'True', 1, 4]], ['Value', 'yes', ['Attribute', 'att0', ['Value', 'Junior', ['Leaf', 'False', 2, 3]], ['Value', 'Mid', ['Leaf', 'True', 1, 3]], ['Value', 'Senior', ['Leaf', 'False', 0, 3]]]]]]]],
                             [['Attribute', 'att3', ['Value', 'no', ['Attribute', 'att2', ['Value', 'no', ['Attribute', 'att0', ['Value', 'Junior', ['Leaf', 'True', 3, 7]], ['Value', 'Mid', ['Leaf', 'True', 1, 7]], ['Value', 'Senior', ['Leaf', 'False', 3, 7]]]], ['Value', 'yes', ['Leaf', 'True', 3, 10]]]], ['Value', 'yes', ['Attribute', 'att2', ['Value', 'no', ['Leaf', 'True', 1, 4]], ['Value', 'yes', ['Attribute', 'att0', ['Value', 'Junior', ['Leaf', 'False', 2, 3]], ['Value', 'Mid', ['Leaf', 'True', 1, 3]], ['Value', 'Senior', ['Leaf', 'False', 0, 3]]]]]]]],
                             [['Attribute', 'att3', ['Value', 'no', ['Attribute', 'att2', ['Value', 'no', ['Attribute', 'att0', ['Value', 'Junior', ['Leaf', 'True', 3, 7]], ['Value', 'Mid', ['Leaf', 'True', 1, 7]], ['Value', 'Senior', ['Leaf', 'False', 3, 7]]]], ['Value', 'yes', ['Leaf', 'True', 3, 10]]]], ['Value', 'yes', ['Attribute', 'att2', ['Value', 'no', ['Leaf', 'True', 1, 4]], ['Value', 'yes', ['Attribute', 'att0', ['Value', 'Junior', ['Leaf', 'False', 2, 3]], ['Value', 'Mid', ['Leaf', 'True', 1, 3]], ['Value', 'Senior', ['Leaf', 'False', 0, 3]]]]]]]],
                             [['Attribute', 'att3', ['Value', 'no', ['Attribute', 'att2', ['Value', 'no', ['Attribute', 'att0', ['Value', 'Junior', ['Leaf', 'True', 3, 7]], ['Value', 'Mid', ['Leaf', 'True', 1, 7]], ['Value', 'Senior', ['Leaf', 'False', 3, 7]]]], ['Value', 'yes', ['Leaf', 'True', 3, 10]]]], ['Value', 'yes', ['Attribute', 'att2', ['Value', 'no', ['Leaf', 'True', 1, 4]], ['Value', 'yes', ['Attribute', 'att0', ['Value', 'Junior', ['Leaf', 'False', 2, 3]], ['Value', 'Mid', ['Leaf', 'True', 1, 3]], ['Value', 'Senior', ['Leaf', 'False', 0, 3]]]]]]]]]
    interview_trees.headers = [['att0', 'att1', 'att2', 'att3'], ['att0', 'att1', 'att2', 'att3'], ['att0', 'att1', 'att2', 'att3'], ['att0', 'att1', 'att2', 'att3'], ['att0', 'att1', 'att2', 'att3'], ['att0', 'att1', 'att2', 'att3'], ['att0', 'att1', 'att2', 'att3']]"""
    header_interview = ["level", "lang", "tweets", "phd", "interviewed_well"]
    X_test_interview = [["Junior", "Java", "yes", "no"], ["Junior", "Java", "yes", "yes"]]
    X_train_interview = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]
    y_train_interview = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]
    interview_trees = MyRandomForestClassifier()
    interview_trees.fit(X_train_interview, y_train_interview, 20, 2, 7)
    y_pred = interview_trees.predict(X_test_interview)
    assert y_pred == ["True", "False"]

    
def test_naive_bayes_classifier_fit():
    # in-class Naive Bayes example (lab task #1)
    header_inclass_example = ["att1", "att2"]
    X_train_inclass_example = [
        [1, 5], # yes
        [2, 6], # yes
        [1, 5], # no
        [1, 5], # no
        [1, 6], # yes
        [2, 6], # no
        [1, 5], # yes
        [1, 6] # yes
    ]
    y_train_inclass_example = ["yes", "yes", "no", "no", "yes", "no", "yes", "yes"]
    class_example_priors = {'yes': [5, 8], 'no': [3, 8]}
    class_example_posteriors = {'yes': {'att1': {'1': [4, 5], '2': [1, 5]}, 
                                        'att2': {'5': [2, 5], '6': [3, 5]}},
                                        'no': {'att1': {'1': [2, 3], '2': [1, 3]},
                                               'att2': {'5': [2, 3], '6': [1, 3]}}}
    classifier1 = MyNaiveBayesClassifier()
    classifier1.fit(X_train_inclass_example, y_train_inclass_example)
    assert classifier1.priors == class_example_priors
    assert classifier1.posteriors == class_example_posteriors

    # MA7 (fake) iPhone purchases dataset
    header_iphone = ["standing", "job_status", "credit_rating"]
    X_train_iphone = [
        [1, 3, "fair"],
        [1, 3, "excellent"],
        [2, 3, "fair"],
        [2, 2, "fair"],
        [2, 1, "fair"],
        [2, 1, "excellent"],
        [2, 1, "excellent"],
        [1, 2, "fair"],
        [1, 1, "fair"],
        [2, 2, "fair"],
        [1, 2, "excellent"],
        [2, 2, "excellent"],
        [2, 3, "fair"],
        [2, 2, "excellent"],
        [2, 3, "fair"]
    ]
    y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]
    iphone_priors = {'yes': [10, 15], 'no': [5, 15]}
    iphone_posteriors = {'yes': {'att1': {'1': [2, 10], '2': [8, 10]}, 
                                 'att2': {'1': [3, 10], '2': [4, 10], '3': [3, 10]}, 
                                 'att3': {'fair': [7, 10], 'excellent': [3, 10]}},
                                 'no': {'att1': {'1': [3, 5], '2': [2, 5]}, 
                                        'att2': {'1': [1, 5], '2': [2, 5], '3': [2, 5]},
                                          'att3': {'fair': [2, 5], 'excellent': [3, 5]}}}
    classifier2 = MyNaiveBayesClassifier()
    classifier2.fit(X_train_iphone, y_train_iphone)
    assert iphone_priors == classifier2.priors
    assert iphone_posteriors == classifier2.posteriors

    # Bramer 3.2 train dataset
    header_train = ["day", "season", "wind", "rain"]
    X_train_train = [
        ["weekday", "spring", "none", "none"],
        ["weekday", "winter", "none", "slight"],
        ["weekday", "winter", "none", "slight"],
        ["weekday", "winter", "high", "heavy"],
        ["saturday", "summer", "normal", "none"],
        ["weekday", "autumn", "normal", "none"],
        ["holiday", "summer", "high", "slight"],
        ["sunday", "summer", "normal", "none"],
        ["weekday", "winter", "high", "heavy"],
        ["weekday", "summer", "none", "slight"],
        ["saturday", "spring", "high", "heavy"],
        ["weekday", "summer", "high", "slight"],
        ["saturday", "winter", "normal", "none"],
        ["weekday", "summer", "high", "none"],
        ["weekday", "winter", "normal", "heavy"],
        ["saturday", "autumn", "high", "slight"],
        ["weekday", "autumn", "none", "heavy"],
        ["holiday", "spring", "normal", "slight"],
        ["weekday", "spring", "normal", "none"],
        ["weekday", "spring", "normal", "slight"]
    ]
    y_train_train = ["on time", "on time", "on time", "late", "on time", "very late", "on time",
                    "on time", "very late", "on time", "cancelled", "on time", "late", "on time",
                    "very late", "on time", "on time", "on time", "on time", "on time"]
    train_train_priors = {'on time': [14, 20], 'late': [2, 20], 'very late': [3, 20], 'cancelled': [1, 20]}
    train_train_posteriors = {'on time': {'att1': {'weekday': [9, 14], 'saturday': [2, 14], 'sunday': [1, 14], 'holiday': [2, 14]},
                                            'att2': {'spring': [4, 14], 'summer': [6, 14], 'autumn': [2, 14], 'winter': [2, 14]},
                                            'att3': {'none': [5, 14], 'high': [4, 14], 'normal': [5, 14]},
                                            'att4': {'none': [5, 14], 'slight': [8, 14], 'heavy': [1, 14]}},
                                            'late': {'att1': {'weekday': [1, 2], 'saturday': [1, 2]},
                                                     'att2': {'winter': [2, 2]},
                                                     'att3': {'high': [1, 2], 'normal': [1, 2]},
                                                     'att4': {'none': [1, 2], 'heavy': [1, 2]}},
                                                     'very late': {'att1': {'weekday': [3, 3]},
                                                                   'att2': {'autumn': [1, 3], 'winter': [2, 3]},
                                                                   'att3': {'high': [1, 3], 'normal': [2, 3]},
                                                                   'att4': {'none': [1, 3], 'heavy': [2, 3]}},
                                                                   'cancelled': {'att1': {'saturday': [1, 1]},
                                                                                 'att2': {'spring': [1, 1]},
                                                                                 'att3': {'high': [1, 1]},
                                                                                'att4': {'heavy': [1, 1]}}}
    classifier3 = MyNaiveBayesClassifier()
    classifier3.fit(X_train_train, y_train_train)
    print(classifier3.posteriors)
    assert train_train_priors == classifier3.priors
    assert train_train_posteriors == classifier3.posteriors


def test_naive_bayes_classifier_predict():
    # in-class Naive Bayes example (lab task #1)
    header_inclass_example = ["att1", "att2"]
    X_train_inclass_example = [
        [1, 5], # yes
        [2, 6], # yes
        [1, 5], # no
        [1, 5], # no
        [1, 6], # yes
        [2, 6], # no
        [1, 5], # yes
        [1, 6] # yes
    ]
    y_train_inclass_example = ["yes", "yes", "no", "no", "yes", "no", "yes", "yes"]
    in_class_pred = MyNaiveBayesClassifier()
    in_class_pred.fit(X_train_inclass_example, y_train_inclass_example)
    unseen_instance = [[1,5]]
    test_classifier = in_class_pred.predict(unseen_instance)
    assert test_classifier == ["yes"]
    # MA7 (fake) iPhone purchases dataset
    header_iphone = ["standing", "job_status", "credit_rating"]
    X_train_iphone = [
        [1, 3, "fair"],
        [1, 3, "excellent"],
        [2, 3, "fair"],
        [2, 2, "fair"],
        [2, 1, "fair"],
        [2, 1, "excellent"],
        [2, 1, "excellent"],
        [1, 2, "fair"],
        [1, 1, "fair"],
        [2, 2, "fair"],
        [1, 2, "excellent"],
        [2, 2, "excellent"],
        [2, 3, "fair"],
        [2, 2, "excellent"],
        [2, 3, "fair"]
    ]
    y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]
    iphone_class = MyNaiveBayesClassifier()
    iphone_class.fit(X_train_iphone, y_train_iphone)
    unseen_instances = [[2,2,"fair"],[1,1,"excellent"]]
    iphone_pred = iphone_class.predict(unseen_instances)
    assert iphone_pred == ["yes","no"]
    
    # Bramer 3.2 train dataset
    header_train = ["day", "season", "wind", "rain"]
    X_train_train = [
        ["weekday", "spring", "none", "none"],
        ["weekday", "winter", "none", "slight"],
        ["weekday", "winter", "none", "slight"],
        ["weekday", "winter", "high", "heavy"],
        ["saturday", "summer", "normal", "none"],
        ["weekday", "autumn", "normal", "none"],
        ["holiday", "summer", "high", "slight"],
        ["sunday", "summer", "normal", "none"],
        ["weekday", "winter", "high", "heavy"],
        ["weekday", "summer", "none", "slight"],
        ["saturday", "spring", "high", "heavy"],
        ["weekday", "summer", "high", "slight"],
        ["saturday", "winter", "normal", "none"],
        ["weekday", "summer", "high", "none"],
        ["weekday", "winter", "normal", "heavy"],
        ["saturday", "autumn", "high", "slight"],
        ["weekday", "autumn", "none", "heavy"],
        ["holiday", "spring", "normal", "slight"],
        ["weekday", "spring", "normal", "none"],
        ["weekday", "spring", "normal", "slight"]
    ]
    y_train_train = ["on time", "on time", "on time", "late", "on time", "very late", "on time",
                     "on time", "very late", "on time", "cancelled", "on time", "late", "on time",
                     "very late", "on time", "on time", "on time", "on time", "on time"]
    bramer_example = MyNaiveBayesClassifier()
    bramer_example.fit(X_train_train, y_train_train)
    bramer_unseen1 = [["weekday","winter","high","heavy"]]
    bramer_ex_1 = bramer_example.predict(bramer_unseen1)
    assert bramer_ex_1 == ["very late"]
    bramer_unseen2 = [["weekday", "summer", "high","heavy"],["sunday", "summer", "normal", "slight"]]
    bramer_ex_2 = bramer_example.predict(bramer_unseen2)
    assert bramer_ex_2 == ["on time", "on time"]

def test_kneighbors_classifier_kneighbors():
    # from in-class #1  (4 instances)
    X_train_class_example1 = [[1, 1], [1, 0], [0.33, 0], [0, 0]]
    y_train_class_example1 = ["bad", "bad", "good", "good"]

    four_inst = MyKNeighborsClassifier()
    four_inst.fit(X_train_class_example1, y_train_class_example1)
    dist_ex1, neigh_ex1 = four_inst.kneighbors([[0.33, 1]])
    assert np.allclose(dist_ex1, [[0.67, 1.0, 1.053]], atol=1e-2)
    assert np.allclose(neigh_ex1, [[0, 2, 3]], atol=1e-2)
    # from in-class #2 (8 instances)
    # assume normalized
    X_train_class_example2 = [
            [3, 2],
            [6, 6],
            [4, 1],
            [4, 4],
            [1, 2],
            [2, 0],
            [0, 3],
            [1, 6]]

    y_train_class_example2 = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]
    eight_inst = MyKNeighborsClassifier()
    eight_inst.fit(X_train_class_example2, y_train_class_example2)
    dist_ex2, neigh_ex2 = eight_inst.kneighbors([[2, 3]])
    assert np.allclose(dist_ex2,[[1.414, 1.414, 2.0]], atol=1e-2)
    assert np.allclose(neigh_ex2, [[0, 4, 6]], atol=1e-2)
    # from Bramer
    header_bramer_example = ["Attribute 1", "Attribute 2"]
    X_train_bramer_example = [
        [0.8, 6.3],
        [1.4, 8.1],
        [2.1, 7.4],
        [2.6, 14.3],
        [6.8, 12.6],
        [8.8, 9.8],
        [9.2, 11.6],
        [10.8, 9.6],
        [11.8, 9.9],
        [12.4, 6.5],
        [12.8, 1.1],
        [14.0, 19.9],
        [14.2, 18.5],
        [15.6, 17.4],
        [15.8, 12.2],
        [16.6, 6.7],
        [17.4, 4.5],
        [18.2, 6.9],
        [19.0, 3.4],
        [19.6, 11.1]]

    y_train_bramer_example = ["-", "-", "-", "+", "-", "+", "-", "+", "+", "+", "-", "-", "-",\
            "-", "-", "+", "+", "+", "-", "+"]
    bramer_inst = MyKNeighborsClassifier(n_neighbors=5)
    bramer_inst.fit(X_train_bramer_example, y_train_bramer_example)
    dist_ex3, neigh_ex3 = bramer_inst.kneighbors([[9.1, 11.0]])
    assert np.allclose(dist_ex3, [[0.608, 1.237, 2.202, 2.802, 2.915]], atol=1e-2)
    assert np.allclose(neigh_ex3, [[6, 5, 7, 4, 8]], atol=1e-2)

def test_kneighbors_classifier_predict():
    # from in-class #1  (4 instances)
    X_train_class_example1 = [[1, 1], [1, 0], [0.33, 0], [0, 0]]
    y_train_class_example1 = ["bad", "bad", "good", "good"]

    four_inst = MyKNeighborsClassifier()
    four_inst.fit(X_train_class_example1, y_train_class_example1)
    y_pred_ex1 = four_inst.predict([[0.33, 1]])
    assert y_pred_ex1 == ["good"]
    # from in-class #2 (8 instances)
    # assume normalized
    X_train_class_example2 = [
            [3, 2],
            [6, 6],
            [4, 1],
            [4, 4],
            [1, 2],
            [2, 0],
            [0, 3],
            [1, 6]]

    y_train_class_example2 = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]
    eight_inst = MyKNeighborsClassifier()
    eight_inst.fit(X_train_class_example2, y_train_class_example2)
    y_pred_ex2 = eight_inst.predict([[2, 3]])
    assert y_pred_ex2 == ["yes"]
    # from Bramer
    header_bramer_example = ["Attribute 1", "Attribute 2"]
    X_train_bramer_example = [
        [0.8, 6.3],
        [1.4, 8.1],
        [2.1, 7.4],
        [2.6, 14.3],
        [6.8, 12.6],
        [8.8, 9.8],
        [9.2, 11.6],
        [10.8, 9.6],
        [11.8, 9.9],
        [12.4, 6.5],
        [12.8, 1.1],
        [14.0, 19.9],
        [14.2, 18.5],
        [15.6, 17.4],
        [15.8, 12.2],
        [16.6, 6.7],
        [17.4, 4.5],
        [18.2, 6.9],
        [19.0, 3.4],
        [19.6, 11.1]]

    y_train_bramer_example = ["-", "-", "-", "+", "-", "+", "-", "+", "+", "+", "-", "-", "-",\
            "-", "-", "+", "+", "+", "-", "+"]
    bramer_inst = MyKNeighborsClassifier(n_neighbors=5)
    bramer_inst.fit(X_train_bramer_example, y_train_bramer_example)
    y_pred_ex3 = bramer_inst.predict([[9.1, 11.0]])
    assert y_pred_ex3 == ["+"] 
