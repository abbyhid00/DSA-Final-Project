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

from analysis_code.myclassifiers import MyDecisionTreeClassifier,\
    MyNaiveBayesClassifier,\
    MyKNeighborsClassifier,\
    RandomForestClassifier
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
    
def test_ranom_forest_classifier_predict():
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


def test_decision_tree_classifier_fit():
    #test case 1
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
    tree_interview = \
            ["Attribute", "att0",
                ["Value", "Junior", 
                    ["Attribute", "att3",
                        ["Value", "no", 
                            ["Leaf", "True", 3, 5]
                       ],
                        ["Value", "yes", 
                            ["Leaf", "False", 2, 5]
                        ]
                  ]
                ],
                ["Value", "Mid",
                    ["Leaf", "True", 4, 14]
                ],
                ["Value", "Senior",
                    ["Attribute", "att2",
                        ["Value", "no",
                            ["Leaf", "False", 3, 5]
                        ],
                        ["Value", "yes",
                            ["Leaf", "True", 2, 5]
                        ]
                    ]
                ]
            ]
    interview_gen_tree = MyDecisionTreeClassifier()
    interview_gen_tree.fit(X_train_interview,y_train_interview)
    assert interview_gen_tree.tree == tree_interview
    
    #test case 2 with MA7
    header_MA7 = ["standing", "job_status", "credit_rating", "buys_iphone"]
    X_train_MA7 = [
        ["1","3","fair"],
        ["1","3","excellent"],
        ["2","3","fair"],
        ["2","2","fair"],
        ["2","1","fair"],
        ["2","1","excellent"],
        ["2","1","excellent"],
        ["1","2","fair"],
        ["1","1","fair"],
        ["2","2","fair"],
        ["1","2","excellent"],
        ["2","2","excellent"],
        ["2","3","fair"],
        ["2","2","excellent"],
        ["2","3","fair"]
    ]
    y_train_MA7 = ["no","no", "yes","yes","yes","no", "yes", "no","yes","yes","yes","yes","yes","no","yes"]
    tree_MA7 = \
            ["Attribute", "att0",
                ["Value", "1", 
                    ["Attribute", "att1",
                        ["Value", "1",
                            ["Leaf", "yes",1, 5]
                        ],
                        ["Value", "2",
                            ["Attribute", "att2",
                                ["Value","excellent",
                                    ["Leaf","yes",1,2]
                                ],
                                ["Value", "fair",
                                    ["Leaf","no",1,2]
                                ]
                            ]
                        ],
                        ["Value","3",
                            ["Leaf","no",2,5]
                        ]
                    ]
                ],
                ["Value", "2",
                    ["Attribute","att2",
                        ["Value", "excellent",
                            ["Attribute", "att1",
                                ["Value", "1",
                                    ["Leaf","no",2,4]
                                ],
                                ["Value","2",
                                    ["Leaf","no",2,4]
                                ],
                                ["Value", "3",
                                    ["Leaf", "no",0,4]
                                ]
                            ]
                        ],
                        ["Value", "fair",
                            ["Leaf","yes",6,10]
                        ]
                    ]
                ]
            ]
    MA7_tree = MyDecisionTreeClassifier()
    MA7_tree.fit(X_train_MA7,y_train_MA7)
    assert MA7_tree.tree == tree_MA7

def test_decision_tree_classifier_predict():
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
    tree_interview = \
            ["Attribute", "att0",
                ["Value", "Junior", 
                    ["Attribute", "att3",
                        ["Value", "no", 
                            ["Leaf", "True", 3, 5]
                       ],
                        ["Value", "yes", 
                            ["Leaf", "False", 2, 5]
                        ]
                  ]
                ],
                ["Value", "Mid",
                    ["Leaf", "True", 4, 14]
                ],
                ["Value", "Senior",
                    ["Attribute", "att2",
                        ["Value", "no",
                            ["Leaf", "False", 3, 5]
                        ],
                        ["Value", "yes",
                            ["Leaf", "True", 2, 5]
                        ]
                    ]
                ]
            ]
    interview_gen_tree = MyDecisionTreeClassifier()
    interview_gen_tree.fit(X_train_interview,y_train_interview)
    X_test = [["Junior", "Java", "yes", "no"]]
    X_test2 = [["Junior", "Java", "yes", "yes"]]
    pred_val = interview_gen_tree.predict(X_test)
    interview_gen_tree.print_decision_rules()
    assert  pred_val == ["True"] #test instance 1
    pred_val2 = interview_gen_tree.predict(X_test2)
    assert pred_val2 == ["False"]
    
    #test case 2 with MA7
    header_MA7 = ["standing", "job_status", "credit_rating", "buys_iphone"]
    X_train_MA7 = [
        ["1","3","fair"],
        ["1","3","excellent"],
        ["2","3","fair"],
        ["2","2","fair"],
        ["2","1","fair"],
        ["2","1","excellent"],
        ["2","1","excellent"],
        ["1","2","fair"],
        ["1","1","fair"],
        ["2","2","fair"],
        ["1","2","excellent"],
        ["2","2","excellent"],
        ["2","3","fair"],
        ["2","2","excellent"],
        ["2","3","fair"]
    ]
    y_train_MA7 = ["no","no", "yes","yes","yes","no", "yes", "no","yes","yes","yes","yes","yes","no","yes"]
    tree_MA7 = \
            ["Attribute", "att0",
                ["Value", "1", 
                    ["Attribute", "att1",
                        ["Value", "1",
                            ["Leaf", "yes",1, 5]
                        ],
                        ["Value", "2",
                            ["Attribute", "att2",
                                ["Value","excellent",
                                    ["Leaf","yes",1,2]
                                ],
                                ["Value", "fair",
                                    ["Leaf","no",1,2]
                                ]
                            ]
                        ],
                        ["Value","3",
                            ["Leaf","no",2,5]
                        ]
                    ]
                ],
                ["Value", "2",
                    ["Attribute","att2",
                        ["Value", "excellent",
                            ["Attribute", "att1",
                                ["Value", "1",
                                    ["Leaf","no",2,4]
                                ],
                                ["Value","2",
                                    ["Leaf","no",2,4]
                                ],
                                ["Value", "3",
                                    ["Leaf", "no",0,4]
                                ]
                            ]
                        ],
                        ["Value", "fair",
                            ["Leaf","yes",6,10]
                        ]
                    ]
                ]
            ]
    unseen_instances = [[2,2,"fair"],[1,1,"excellent"]]
    iphone_class = MyDecisionTreeClassifier()
    iphone_class.fit(X_train_MA7, y_train_MA7)
    iphone_pred = iphone_class.predict(unseen_instances)
    iphone_class.print_decision_rules(header_MA7)
    assert iphone_pred == ["yes","yes"]
    
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
    naive_class = MyNaiveBayesClassifier()
    naive_class.fit(X_train_inclass_example, y_train_inclass_example) # no need to pass in labels here
    in_class_posteriors = {
        "att1"  : {
            1 : [4/5,2/3],
            2 : [1/5,1/3],
        },
        "att2" : {
            5 : [2/5,2/3],
            6 : [3/5,1/3]
        }
    }
    in_class_priors = {
        "yes" : 5/8,
        "no" : 3/8
    }
    assert in_class_posteriors == naive_class.posteriors
    assert in_class_priors == naive_class.priors
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
    ma7_naive = MyNaiveBayesClassifier()
    ma7_naive.fit(X_train_iphone, y_train_iphone)
    ma7_posteriors = {
        "att1"  : { # need to come back and look at these values 
            1 : [3/5,2/10],
            2 : [2/5,8/10]
        },
        "att2" : {
            1 : [1/5,3/10],
            2 : [2/5,4/10],
            3 : [2/5,3/10]
        },
        "att3" : {
            "fair" : [2/5,7/10],
            "excellent" : [3/5,3/10]
        }
    }
    ma7_priors = {
        "yes" : 2/3,
        "no" : 1/3
    }
    assert ma7_posteriors == ma7_naive.posteriors
    assert ma7_priors == ma7_naive.priors
    
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
    bramer_naive = MyNaiveBayesClassifier()
    bramer_naive.fit(X_train_train, y_train_train)
    bramer_posteriors = {
        "att1"  : {
            "weekday" : [9/14,1/2,3/3,0/1],
            "saturday" : [2/14,1/2,0/3,1/1],
            "sunday" : [1/14,0.0,0.0,0.0],
            "holiday" : [2/14,0,0,0],
        },
        "att2" : {
            "spring" : [4/14,0,0,1],
            "summer" : [6/14,0,0,0],
            "autumn" : [2/14,0,1/3,0],
            "winter" : [2/14,2/2,2/3,0],
        },
        "att3" : {
            "none" : [5/14,0,0,0],
            "high" : [4/14,1/2,1/3,1],
            "normal" : [5/14,1/2,2/3,0],
        },
        "att4" : {
            "none" : [5/14,1/2,1/3,0],
            "slight" : [8/14,0,0,0],
            "heavy" : [1/14,1/2,2/3,1],
        }
    }
    bramer_priors = {
        "on time" : 14/20,
        "late" : 2/20,
        "very late" : 3/20,
        "cancelled" : 1/20
    }
    assert bramer_posteriors == bramer_naive.posteriors
    assert bramer_priors == bramer_naive.priors

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
    example1_KNN = MyKNeighborsClassifier()
    example1_KNN.fit(X_train_class_example1, y_train_class_example1)
    x_test_example1 =[[0.33, 1]]
    distances1, neighbors1 = example1_KNN.kneighbors(x_test_example1)
    desk_check_distances = [[0.67],[1],[1.053]]
    desk_check_neighbors = [[0],[2],[3]]
    assert np.allclose(distances1, desk_check_distances, atol=1e-03)
    assert np.allclose(neighbors1, desk_check_neighbors, atol=1e-03)

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
    example2_KNN = MyKNeighborsClassifier()
    example2_KNN.fit(X_train_class_example2, y_train_class_example2)
    x_test_example2 = [[2,3]]
    distances2, neighbors2 = example2_KNN.kneighbors(x_test_example2)
    desk_check_distances2 = [[1.4142135623730951],[1.4142135623730951],[2.0]]
    desk_check_neighbors2 = [[0],[4],[6]]
    assert np.allclose(distances2, desk_check_distances2)
    assert np.allclose(neighbors2, desk_check_neighbors2)
    
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
    example3_KNN = MyKNeighborsClassifier(n_neighbors=5)
    example3_KNN.fit(X_train_bramer_example, y_train_bramer_example)
    x_test_example3 = [[9.1,11.0]]
    distance3, neighbors3 = example3_KNN.kneighbors(x_test_example3)
    desk_check_distances3 = [[0.608],[1.237],[2.202],[2.802],[2.915]]
    desk_check_neighbors3 = [[6],[5],[7],[4],[8]]
    assert np.allclose(distance3, desk_check_distances3,atol=1e-03)
    assert np.allclose(neighbors3, desk_check_neighbors3)

def test_kneighbors_classifier_predict():
    #from class #1
    X_train_class_example1 = [[1, 1], [1, 0], [0.33, 0], [0, 0]]
    y_train_class_example1 = ["bad", "bad", "good", "good"]
    example1_KNN = MyKNeighborsClassifier()
    example1_KNN.fit(X_train_class_example1, y_train_class_example1)
    x_test_example1 =[[0.33,1]]
    y_pred = example1_KNN.predict(x_test_example1)
    desk_check_pred = ["good"]
    assert y_pred == desk_check_pred

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
    example2_KNN = MyKNeighborsClassifier()
    example2_KNN.fit(X_train_class_example2, y_train_class_example2)
    x_test_example2 =[[2,3]]
    y_pred2 = example2_KNN.predict(x_test_example2)
    desk_check_pred2 = ["yes"]
    assert y_pred2 == desk_check_pred2 

    #case 3
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
    example3_KNN = MyKNeighborsClassifier(n_neighbors=5)
    example3_KNN.fit(X_train_bramer_example, y_train_bramer_example)
    x_test_example3 = [[9.1,11.0]]
    y_pred3 = example3_KNN.predict(x_test_example3)
    desk_check_pred3 = ["+"]
    assert y_pred3 == desk_check_pred3

def test_dummy_classifier_fit():
    # case 1
    np.random.seed(0)
    X_train = [[val] for val in list(range(0, 100))]
    y_train = list(np.random.choice(["yes", "no"], 100, replace=True, p=[0.7, 0.3]))
    dummy_case1 = MyDummyClassifier()
    dummy_case1.fit(X_train, y_train)
    assert dummy_case1.most_common_label == "yes"

    # case 2
    X_train2 = [[val] for val in list(range(0, 100))]
    y_train2 = list(np.random.choice(["yes", "no", "maybe"], 100, replace=True, p=[0.2, 0.6, 0.2]))
    dummy_case2 = MyDummyClassifier()
    dummy_case2.fit(X_train2, y_train2)
    assert dummy_case2.most_common_label == "no"

    #case 3
    X_train3 = [[val] for val in list(range(0, 100))]
    y_train3 = list(np.random.choice(["absolutely", "no", "not exactly"], 100, replace=True, p=[0.3, 0.3, 0.4]))
    dummy_case3 = MyDummyClassifier()
    dummy_case3.fit(X_train3, y_train3)
    assert dummy_case3.most_common_label == "not exactly"

def test_dummy_classifier_predict():
     # case 1
    X_train = [[val] for val in list(range(0, 100))]
    y_train = list(np.random.choice(["yes", "no"], 100, replace=True, p=[0.7, 0.3]))
    dummy_case1 = MyDummyClassifier()
    x_test1 = [[101]]
    dummy_case1.fit(X_train, y_train)
    y_pred1 = dummy_case1.predict(x_test1)
    assert np.array_equal(y_pred1, ["yes"])

    # case 2
    X_train2 = [[val] for val in list(range(0, 100))]
    y_train2 = list(np.random.choice(["yes", "no", "maybe"], 100, replace=True, p=[0.2, 0.6, 0.2]))
    dummy_case2 = MyDummyClassifier()
    dummy_case2.fit(X_train2, y_train2)
    x_test2 = [[105]]
    y_pred2 = dummy_case2.predict(x_test2)
    assert np.array_equal(y_pred2, ["no"])

    #case 3
    X_train3 = [[val] for val in list(range(0, 100))]
    y_train3 = list(np.random.choice(["absolutely", "no", "not exactly"], 100, replace=True, p=[0.3, 0.3, 0.4]))
    dummy_case3 = MyDummyClassifier()
    dummy_case3.fit(X_train3, y_train3)
    x_test3 = [[110]]
    y_pred3 = dummy_case3.predict(x_test3)
    assert np.array_equal(y_pred3, ["not exactly"])
