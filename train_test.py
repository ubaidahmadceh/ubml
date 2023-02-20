def metric_calc(metric_name, actual_data_train, pred_data_train, actual_data_test, pred_data_test):
    """
    This function calculates the metric score

    Inputs: 
        - metric_name (str): e.g r2_score from sklearn.metrics 

        - actual_data_train (output from sklearn train_test_split()): x_train
        - pred_data_train (output or model.predict(x_train)): y_pred_train
            These will be used to compute metric score on training set
        - actual_data_test (output from sklearn train_test_split()): x_test
        - pred_data_test (output or model.predict(x_test)): y_pred_test 
            These will be used to compute metric score on testing set
    Return:
        - computed metric score on traing and testing data (list): [result_train, result_test] 
    """
    # importing importlib to dynamically import the metrics from sklearn
    import importlib        
    # creating special rule for RMSE as its not available directly in sklearn.metrics
    if metric_name == "root_mean_squared_error":
        # import mean_squared_error from sklearn.metrics
        metric = getattr(importlib.import_module("sklearn.metrics"), "mean_squared_error")
        # calculating RMSE on training and testing predictions by doing sqaured=False on MSE
        result_train = metric(actual_data_train, pred_data_train, squared=False)
        result_test = metric(actual_data_test, pred_data_test, squared=False)
    else:
        # importing different metrics (it will run in a loop , every iteration will change the metric name)
        metric = getattr(importlib.import_module("sklearn.metrics"), metric_name)
        result_train = metric(actual_data_train, pred_data_train)
        result_test = metric(actual_data_test, pred_data_test)
    return [result_train, result_test]



def regression_train_test(x_train, y_train, x_test, y_test, metric="r2_score", export_best=None, export_model=None, path=None):
    """
    This function will perform regression

    Inputs: 
        - x_train (output from sklearn train_test_split()): x_train
        - y_train (output from sklearn train_test_split()): y_train
        - x_test (output from sklearn train_test_split()): x_test
        - y_test (output from sklearn train_test_split()): y_test
        - metric (str): Give it a metric name for which you want to choose best model
            By default it will take value as r2_score and choose the best model with the highest validation r2_score, but you can change the metric
        - export_best (bool): set it as True if you want to export the best model as a pickle file 
        - export_model (str): give it a model name if you want to export that specific model's pickle file
            models can be selected as follows:
                Linear Regression
                Lasso Regression
                Ridge Regression
                Support Vector Regression
                Decision Tree Regression
                Random Forest Regression
        - path (str): give a path with model name if you want to specify where to save the model and by what name
    Returns:
        - Metrics Performance Table (type: A Pandas Dataframe)
        - Best Model Name
        - Best Model pickle file (only if you set argument export_best=True)
        - Custom Model pickle file (only if you set argument export_model=True)
    """
    
    # importing regression models
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import Lasso
    from sklearn.linear_model import Ridge
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor

    # initializing models 
    linear_reg = LinearRegression()
    lasso_reg = Lasso()
    ridge_reg = Ridge()
    svr_reg = SVR(kernel = 'rbf')
    decision_tree_reg = DecisionTreeRegressor()
    random_forest_reg = RandomForestRegressor()

    # training models
    models = [linear_reg, lasso_reg, ridge_reg, svr_reg, decision_tree_reg, random_forest_reg]
    for i in models:
        i.fit(x_train, y_train)

    # doing predictions on training data
    train_pred_res = []
    # in first iteration of loop, linear_reg will be used for prediction, in second iteration lasso_reg will be used and son
    for i in models:
        # appending predictions on each model in a list (train_pred_res), Now train_pred_res[0] will predictions of linear regression and train_pred_res[1] will be of lasso and so on 
        train_pred_res.append(i.predict(x_train))

    # doing predictions on testing data
    test_pred_res = []
    for i in models:
        test_pred_res.append(i.predict(x_test))

    # computing metrics
    regression_models_names = ["Linear Regression", "Lasso Regression", "Ridge Regression", "Support Vector Regression", "Decision Tree Regression", "Random Forest Regression"]
    metrics_names = ["r2_score", "mean_squared_error", "root_mean_squared_error", "mean_absolute_error", "explained_variance_score", "max_error"]

    model_pred_counter = 0
    all_models_metrics = []
    for i in range(len(regression_models_names)):
        metrics_result = []
        for i in range(len(metrics_names)):
            # in first iteration, i will be 0 so metrics_names[0] will be r2_score and model_pred_counter will also be 0 so train_pred_res[0] and test_pred_res[0] will be prediction results for linear regression (check prediction step above, if don't understand: basically train_pred_res is a list and train_pred_res[0] is prediction result for linear regression and train_pred_res[1] is for lasso and so on)
            # so the result will be output of r2_score for training and testing, it will be a list as result[0] will be r2_score for training set and result[1] will be r_2 score for testing data 
            # In second iteration i will be 1 so metrics_names[1] will be mean_squared_error so same thing will be done for that and so on for every metric included in above created list metrics_names 
            result = metric_calc(metrics_names[i], y_train, train_pred_res[model_pred_counter], y_test, test_pred_res[model_pred_counter])
            # appending result in a list metrics_result, so now metrics_result[0] will be 
            metrics_result.extend([(format(float(result[0]), "f")), result[1]])
        all_models_metrics.append(metrics_result)
        model_pred_counter += 1

    # making dictionary 
    resultant_dicts = {}   
    for i in range(len(regression_models_names)):
        resultant_dicts[regression_models_names[i]] = all_models_metrics[i]

    row_names = ["r2 score train", 'r2 score test', "mean squared error train", "mean squared error test", "root mean squared error train", "root mean squared error test", "mean absolute error train", "mean absolute error test", "explained variance score train", "explained variance score test", "max error train", "max error test"]

    import pandas as pd
    resultant_df = pd.DataFrame(resultant_dicts, index=row_names)
    resultant_df = resultant_df.transpose().astype(float).round(3)

    metric_test = metric.replace("_", " ") + " test"
    val_acc = []
    for i in range(6):
        val_acc.append(resultant_df.iloc[i][metric_test])
    for i in range(len(val_acc)):
        if val_acc[i] == 1:
            val_acc[i] = -333
    best_model = regression_models_names[val_acc.index(max(val_acc))]

    if export_best:
        import pickle
        model_mapping = {}
        for i in range(len(models)):
            model_mapping[regression_models_names[i]] = models[i]
        if path:
            pickle.dump(model_mapping[best_model], open(path, "wb"))
        else:
            pickle.dump(model_mapping[best_model], open("best_model.pkl", "wb")) 

    if export_model:
        import pickle
        model_mapping = {}
        for i in range(len(models)):
            model_mapping[regression_models_names[i]] = models[i]
        if path:
            pickle.dump(model_mapping[export_model], open(path, "wb"))
        else:
            pickle.dump(model_mapping[export_model], open(export_model.replace(" ", "_") + ".pkl", "wb"))
        

    return resultant_df, best_model



def classification_train_test(x_train, y_train, x_test, y_test, metric="accuracy_score", export_best=None, export_model=None, path=None): 
    """
    This function will perform classification

    Inputs: 
        - x_train (output from sklearn train_test_split()): x_train
        - y_train (output from sklearn train_test_split()): y_train
        - x_test (output from sklearn train_test_split()): x_test
        - y_test (output from sklearn train_test_split()): y_test
        - metric (str): Give it a metric name for which you want to choose best model
            By default it will take value as accuracy_score and choose the best model with the highest validation r2_score, but you can change the metric
        - export_best (bool): set it as True if you want to export the best model as a pickle file 
        - export_model (str): give it a model name if you want to export that specific model's pickle file
            models can be selected as follows:
                Logistic Regression
                KNN Classifier
                SVM Classifier
                Naive Bayes Classifier
                Decision Tree Classifier
                Random Forest Classifier
        - path (str): give a path with model name if you want to specify where to save the model and by what name
    Returns:
        - Metrics Performance Table (type: A Pandas Dataframe)
        - Best Model Name
        - Best Model pickle file (only if you set argument export_best=True)
        - Custom Model pickle file (only if you set argument export_model=True)
    """
    import warnings
    warnings.filterwarnings('ignore')

    # importing regression models
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier

    # initializing models 
    logistic_clsfier = LogisticRegression()
    knn_clsfier = KNeighborsClassifier()
    svc_clsfier = SVC(kernel = 'rbf')
    nb_clsfier = GaussianNB()
    decision_tree_clsfier = DecisionTreeClassifier()
    random_forest_clsfier = RandomForestClassifier()

    models = [logistic_clsfier, knn_clsfier, svc_clsfier, nb_clsfier, decision_tree_clsfier, random_forest_clsfier]

    # training models
    for i in models:
        i.fit(x_train, y_train)

    # doing predictions on training data
    train_pred_res = []
    for i in models:
        train_pred_res.append(i.predict(x_train))

    # doing predictions on testing data
    test_pred_res = []
    for i in models:
        test_pred_res.append(i.predict(x_test))

    # computing metrics
    clsfication_models_names = ["Logistic Regression", "KNN Classifier", "SVM Classifier", "Naive Bayes Classifier", "Decision Tree Classifier", "Random Forest Classifier"]
    metrics_names = ["accuracy_score", "f1_score", "recall_score", "jaccard_score", "precision_score"]

    model_pred_counter = 0 
    all_models_metrics = []
    for i in range(len(clsfication_models_names)):
        metrics_result = []
        for i in range(len(metrics_names)):
            result = metric_calc(metrics_names[i], y_train, train_pred_res[model_pred_counter], y_test, test_pred_res[model_pred_counter])
            metrics_result.extend([(format(float(result[0]), "f")), result[1]])
        all_models_metrics.append(metrics_result)
        model_pred_counter += 1

    # making dictionary 
    resultant_dicts = {}   
    for i in range(len(clsfication_models_names)):
        resultant_dicts[clsfication_models_names[i]] = all_models_metrics[i]

    row_names = ["accuracy score train", 'accuracy score test', "f1 score train", "f1 score test", "recall score train", "recall score test", "jaccard score train", "jaccard score test", "precision score train", "precision score test"]

    import pandas as pd
    resultant_df = pd.DataFrame(resultant_dicts, index=row_names)
    resultant_df = resultant_df.transpose().astype(float).round(3)

    metric_test = metric.replace("_", " ") + " test"
    val_acc = []
    for i in range(6):
        val_acc.append(resultant_df.iloc[i][metric_test])
    for i in range(len(val_acc)):
        if val_acc[i] == 1:
            val_acc[i] = -333
    best_model = clsfication_models_names[val_acc.index(max(val_acc))]

    if export_best:
        import pickle
        model_mapping = {}
        for i in range(len(models)):
            model_mapping[clsfication_models_names[i]] = models[i]
        if path:
            pickle.dump(model_mapping[best_model], open(path, "wb"))
        else:
            pickle.dump(model_mapping[best_model], open("best_model.pkl", "wb")) 

    if export_model:
        import pickle
        model_mapping = {}
        for i in range(len(models)):
            model_mapping[clsfication_models_names[i]] = models[i]
        if path:
            pickle.dump(model_mapping[export_model], open(path, "wb"))
        else:
            pickle.dump(model_mapping[export_model], open(export_model.replace(" ", "_") + ".pkl", "wb"))
        
    return resultant_df, best_model
    



def model_train_test(mode, x_train, y_train, x_test, y_test):
    """
    This function will do regression or classification based on user choice (mode)
    
    Inputs: 
        - mode (str): set it as "regression", if you want to do regression OR set it as "classification", if you want to do classification
        - x_train (output from sklearn train_test_split()): x_train
        - y_train (output from sklearn train_test_split()): y_train
        - x_test (output from sklearn train_test_split()): x_test
        - y_test (output from sklearn train_test_split()): y_test
    Returns:
        - Metrics Performance Table (type: A Pandas Dataframe)
        - Best Model Name
        - Best Model pickle file (only if you set argument export_best=True)
        - Custom Model pickle file (only if you set argument export_model=True)
    """
    if mode == "regression":
        return regression_train_test(x_train, y_train, x_test, y_test)
    elif mode == "classification":
        return classification_train_test(x_train, y_train, x_test, y_test)
    else:
        return "Please specify mode as Regression or classification"
        

def load_model(model):
    import pickle
    mod = pickle.load(open(model, "rb"))
    return mod

def predict(model, input):
    import warnings
    warnings.filterwarnings('ignore')
    import pickle
    mod = pickle.load(open(model, "rb"))
    import numpy as np
    return mod.predict(np.asarray(input).reshape(1,-1))[0]



