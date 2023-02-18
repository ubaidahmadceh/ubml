# new short appraoch
def model_train_test(x_train, y_train, x_test, y_test):
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

    return resultant_df.transpose()

model_train_test(x_train=X_train, y_train=Y_train, x_test=X_test, y_test=Y_test)