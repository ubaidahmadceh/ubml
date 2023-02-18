def metric_calc(metric_name, actual_data_train, pred_data_train, actual_data_test, pred_data_test):
    import importlib        
    if metric_name == "root_mean_squared_error":
        metric = getattr(importlib.import_module("sklearn.metrics"), "mean_squared_error")
        result_train = metric(actual_data_train, pred_data_train, squared=False)
        result_test = metric(actual_data_test, pred_data_test, squared=False)
    else:
        metric = getattr(importlib.import_module("sklearn.metrics"), metric_name)
        result_train = metric(actual_data_train, pred_data_train)
        result_test = metric(actual_data_test, pred_data_test)
    return [result_train, result_test]

# new short appraoch
def regression_train_test(x_train, y_train, x_test, y_test):
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

    models = [linear_reg, lasso_reg, ridge_reg, svr_reg, decision_tree_reg, random_forest_reg]

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
    regression_models_names = ["Linear Regression", "Lasso Regression", "Ridge Regression", "Support Vector Regression", "Decision Tree Regression", "Random Forest Regression"]
    metrics_names = ["r2_score", "mean_squared_error", "root_mean_squared_error", "mean_absolute_error", "explained_variance_score", "max_error"]

    model_pred_counter = 0
    all_models_metrics = []
    for i in range(len(regression_models_names)):
        metrics_result = []
        for i in range(len(metrics_names)):
            result = metric_calc(metrics_names[i], y_train, train_pred_res[model_pred_counter], y_test, test_pred_res[model_pred_counter])
            metrics_result.extend([(format(float(result[0]), "f")), result[1]])
        all_models_metrics.append(metrics_result)
        model_pred_counter += 1

    # making dictionary 
    resultant_dicts = {}   
    for i in range(len(regression_models_names)):
        resultant_dicts[regression_models_names[i]] = all_models_metrics[i]

    row_names = ["r2 score train", 'r2 score test', "MSE train", "MSE test", "RMSE train", "RMSE test", "MAE train", "MAE test", "Explained Variance Score train", "Explained Variance Score test", "Max Error train", "Max Error test"]

    import pandas as pd
    resultant_df = pd.DataFrame(resultant_dicts, index=row_names)

    return resultant_df.transpose().astype(float).round(3)

def classification_train_test(x_train, y_train, x_test, y_test):
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

    row_names = ["Accuracy Score train", 'Accuracy Score test', "F1 Score train", "F1 Score test", "Recall Score train", "Recall Score test", "Jaccard Score train", "Jaccard Score test", "Precision Score train", "Precision Score test"]

    import pandas as pd
    resultant_df = pd.DataFrame(resultant_dicts, index=row_names)
    

    return resultant_df.transpose().astype(float).round(3)