def classification(train_data_df,test_data_df,classifier):

    if classifier=="LogisticRegression":
        clf = sklearn.linear_model.LogisticRegression(penalty='l2')

    if classifier == "naiveBayes":
        clf = sklearn.naive_bayes.BernoulliNB()

    if classifier == "bag":
        tree = sklearn.tree.DecisionTreeClassifier(max_depth=10) #Create an instance of our decision tree classifier.
        clf = sklearn.ensemble.BaggingClassifier(tree, n_estimators=100, max_samples=0.8, random_state=1) 
    
    if classifier == "SVM":
        clf = sklearn.svm.SVC(kernel='linear', probability = False)
    
    if classifier == "NeuralNet":
        clf = sklearn.neural_network.MLPClassifier()
    
    clf.fit(np.stack(train_data_df['vect'], axis=0), train_data_df['category'])
    print(classifier)
    print("Training Accuracy:")
    print(clf.score(np.stack(train_data_df['vect'], axis=0), train_data_df['category']))
    print("Testing Accuracy:")
    print(clf.score(np.stack(test_data_df['vect'], axis=0), test_data_df['category']))
    print("\n")

    return clf

def evaluation(classifier_name, classifier, test_data_df,true_cat):
    # predict
    test_data_df['predict'] = classifier.predict(np.stack(test_data_df['vect'], axis=0))

    # precision, recall, f1 score
    print(classifier_name)
    print("Precision:")
    print(sklearn.metrics.precision_score(test_data_df['category'], test_data_df['predict']))
    print("Recall:")
    print(sklearn.metrics.recall_score(test_data_df['category'], test_data_df['predict']))
    print("F1 Score:")
    print(sklearn.metrics.f1_score(test_data_df['category'], test_data_df['predict']))

    print("True Category is:",true_cat)
    lucem_illud_2020.plotMultiROC(classifier, test_data_df)
    lucem_illud_2020.plotConfusionMatrix(classifier, test_data_df)
    print(lucem_illud_2020.evaluateClassifier(classifier, test_data_df))   

