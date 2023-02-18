def model_train_test(x_train, y_train, x_test, y_test):
    # importing regression models
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import Lasso
    from sklearn.linear_model import Ridge
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor

    # importing regression metrics
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error 
    from sklearn.metrics import mean_absolute_error 
    from sklearn.metrics import explained_variance_score 
    from sklearn.metrics import max_error 

    # initializing models 
    linear_reg = LinearRegression()
    lasso_reg = Lasso()
    ridge_reg = Ridge()
    svr_reg = SVR(kernel = 'rbf')
    decision_tree_reg = DecisionTreeRegressor()
    random_forest_reg = RandomForestRegressor()

    # training models
    linear_reg.fit(x_train, y_train)
    lasso_reg.fit(x_train, y_train)
    ridge_reg.fit(x_train, y_train)
    svr_reg.fit(x_train, y_train)
    decision_tree_reg.fit(x_train, y_train)
    random_forest_reg.fit(x_train, y_train)

    # doing predictions on training data
    linear_reg_pred_train = linear_reg.predict(x_train)
    lasso_reg_pred_train = lasso_reg.predict(x_train)
    ridge_reg_pred_train = ridge_reg.predict(x_train)
    svr_reg_pred_train = svr_reg.predict(x_train)
    decision_tree_reg_pred_train = decision_tree_reg.predict(x_train)
    random_forest_reg_pred_train = random_forest_reg.predict(x_train)

    # doing predictions on testing data
    linear_reg_pred_test = linear_reg.predict(x_test)
    lasso_reg_pred_test = lasso_reg.predict(x_test)
    ridge_reg_pred_test = ridge_reg.predict(x_test)
    svr_reg_pred_test = svr_reg.predict(x_test)
    decision_tree_reg_pred_test = decision_tree_reg.predict(x_test)
    random_forest_reg_pred_test = random_forest_reg.predict(x_test)  

    # computing metrics on training data
    # calculating r2_score of all models on training data   
    linear_reg_r2_score_train = r2_score(y_train, linear_reg_pred_train)
    lasso_reg_r2_score_train = r2_score(y_train, lasso_reg_pred_train)
    ridge_reg_r2_score_train = r2_score(y_train, ridge_reg_pred_train)
    svr_reg_r2_score_train = r2_score(y_train, svr_reg_pred_train)
    decision_tree_reg_r2_score_train = r2_score(y_train, decision_tree_reg_pred_train)
    random_forest_reg_r2_score_train = r2_score(y_train, random_forest_reg_pred_train)

    # calculating mean_squared_error of all models on training data   
    linear_reg_mse_score_train = mean_squared_error(y_train, linear_reg_pred_train)
    lasso_reg_mse_score_train = mean_squared_error(y_train, lasso_reg_pred_train)
    ridge_reg_mse_score_train = mean_squared_error(y_train, ridge_reg_pred_train)
    svr_reg_mse_score_train = mean_squared_error(y_train, svr_reg_pred_train)
    decision_tree_reg_mse_score_train = mean_squared_error(y_train, decision_tree_reg_pred_train)
    random_forest_reg_mse_score_train = mean_squared_error(y_train, random_forest_reg_pred_train)

    # calculating root_mean_squared_error of all models on training data   
    linear_reg_rmse_score_train = mean_squared_error(y_train, linear_reg_pred_train, squared=False)
    lasso_reg_rmse_score_train = mean_squared_error(y_train, lasso_reg_pred_train, squared=False)
    ridge_reg_rmse_score_train = mean_squared_error(y_train, ridge_reg_pred_train, squared=False)
    svr_reg_rmse_score_train = mean_squared_error(y_train, svr_reg_pred_train, squared=False)
    decision_tree_reg_rmse_score_train = mean_squared_error(y_train, decision_tree_reg_pred_train, squared=False)
    random_forest_reg_rmse_score_train = mean_squared_error(y_train, random_forest_reg_pred_train, squared=False)    
    
    # calculating mean_absolute_error of all models on training data   
    linear_reg_mae_score_train = mean_absolute_error(y_train, linear_reg_pred_train)
    lasso_reg_mae_score_train = mean_absolute_error(y_train, lasso_reg_pred_train)
    ridge_reg_mae_score_train = mean_absolute_error(y_train, ridge_reg_pred_train)
    svr_reg_mae_score_train = mean_absolute_error(y_train, svr_reg_pred_train)
    decision_tree_reg_mae_score_train = mean_absolute_error(y_train, decision_tree_reg_pred_train)
    random_forest_reg_mae_score_train = mean_absolute_error(y_train, random_forest_reg_pred_train)
    
    # calculating explained_variance_score of all models on training data   
    linear_reg_evs_score_train = explained_variance_score(y_train, linear_reg_pred_train)
    lasso_reg_evs_score_train = explained_variance_score(y_train, lasso_reg_pred_train)
    ridge_reg_evs_score_train = explained_variance_score(y_train, ridge_reg_pred_train)
    svr_reg_evs_score_train = explained_variance_score(y_train, svr_reg_pred_train)
    decision_tree_reg_evs_score_train = explained_variance_score(y_train, decision_tree_reg_pred_train)
    random_forest_reg_evs_score_train = explained_variance_score(y_train, random_forest_reg_pred_train)
    
    # calculating max_error of all models on training data   
    linear_reg_me_score_train = max_error(y_train, linear_reg_pred_train)
    lasso_reg_me_score_train = max_error(y_train, lasso_reg_pred_train)
    ridge_reg_me_score_train = max_error(y_train, ridge_reg_pred_train)
    svr_reg_me_score_train = max_error(y_train, svr_reg_pred_train)
    decision_tree_reg_me_score_train = max_error(y_train, decision_tree_reg_pred_train)
    random_forest_reg_me_score_train = max_error(y_train, random_forest_reg_pred_train)
    
    # computing metrics on testing data
    # calculating r2_score of all models on testing data   
    linear_reg_r2_score_test = r2_score(y_test, linear_reg_pred_test)
    lasso_reg_r2_score_test = r2_score(y_test, lasso_reg_pred_test)
    ridge_reg_r2_score_test = r2_score(y_test, ridge_reg_pred_test)
    svr_reg_r2_score_test = r2_score(y_test, svr_reg_pred_test)
    decision_tree_reg_r2_score_test = r2_score(y_test, decision_tree_reg_pred_test)
    random_forest_reg_r2_score_test = r2_score(y_test, random_forest_reg_pred_test)

    # calculating mean_squared_error of all models on testing data   
    linear_reg_mse_score_test = mean_squared_error(y_test, linear_reg_pred_test)
    lasso_reg_mse_score_test = mean_squared_error(y_test, lasso_reg_pred_test)
    ridge_reg_mse_score_test = mean_squared_error(y_test, ridge_reg_pred_test)
    svr_reg_mse_score_test = mean_squared_error(y_test, svr_reg_pred_test)
    decision_tree_reg_mse_score_test = mean_squared_error(y_test, decision_tree_reg_pred_test)
    random_forest_reg_mse_score_test = mean_squared_error(y_test, random_forest_reg_pred_test)

    # calculating root_mean_squared_error of all models on testing data   
    linear_reg_rmse_score_test = mean_squared_error(y_test, linear_reg_pred_test, squared=False)
    lasso_reg_rmse_score_test = mean_squared_error(y_test, lasso_reg_pred_test, squared=False)
    ridge_reg_rmse_score_test = mean_squared_error(y_test, ridge_reg_pred_test, squared=False)
    svr_reg_rmse_score_test = mean_squared_error(y_test, svr_reg_pred_test, squared=False)
    decision_tree_reg_rmse_score_test = mean_squared_error(y_test, decision_tree_reg_pred_test, squared=False)
    random_forest_reg_rmse_score_test = mean_squared_error(y_test, random_forest_reg_pred_test, squared=False)    
    
    # calculating mean_absolute_error of all models on testing data   
    linear_reg_mae_score_test = mean_absolute_error(y_test, linear_reg_pred_test)
    lasso_reg_mae_score_test = mean_absolute_error(y_test, lasso_reg_pred_test)
    ridge_reg_mae_score_test = mean_absolute_error(y_test, ridge_reg_pred_test)
    svr_reg_mae_score_test = mean_absolute_error(y_test, svr_reg_pred_test)
    decision_tree_reg_mae_score_test = mean_absolute_error(y_test, decision_tree_reg_pred_test)
    random_forest_reg_mae_score_test = mean_absolute_error(y_test, random_forest_reg_pred_test)
    
    # calculating explained_variance_score of all models on testing data   
    linear_reg_evs_score_test = explained_variance_score(y_test, linear_reg_pred_test)
    lasso_reg_evs_score_test = explained_variance_score(y_test, lasso_reg_pred_test)
    ridge_reg_evs_score_test = explained_variance_score(y_test, ridge_reg_pred_test)
    svr_reg_evs_score_test = explained_variance_score(y_test, svr_reg_pred_test)
    decision_tree_reg_evs_score_test = explained_variance_score(y_test, decision_tree_reg_pred_test)
    random_forest_reg_evs_score_test = explained_variance_score(y_test, random_forest_reg_pred_test)
    
    # calculating max_error of all models on testing data   
    linear_reg_me_score_test = max_error(y_test, linear_reg_pred_test)
    lasso_reg_me_score_test = max_error(y_test, lasso_reg_pred_test)
    ridge_reg_me_score_test = max_error(y_test, ridge_reg_pred_test)
    svr_reg_me_score_test = max_error(y_test, svr_reg_pred_test)
    decision_tree_reg_me_score_test = max_error(y_test, decision_tree_reg_pred_test)
    random_forest_reg_me_score_test = max_error(y_test, random_forest_reg_pred_test)

    # importing pandas dataframe of results
    import pandas as pd

    # creating dataframe for results of training data
    row_names = ["Linear Regression", "Lasso Regression", "Ridge Regression", "Support Vector Regression", "Decision Tree Regression", "Random Forest Regression"]
    data_for_df = {'r2 score train': [format(float(linear_reg_r2_score_train), 'f'), lasso_reg_r2_score_train, ridge_reg_r2_score_train, svr_reg_r2_score_train, decision_tree_reg_r2_score_train, random_forest_reg_r2_score_train],
                   'r2 score test': [format(float(linear_reg_r2_score_test), 'f'), lasso_reg_r2_score_test, ridge_reg_r2_score_test, svr_reg_r2_score_test, decision_tree_reg_r2_score_test, random_forest_reg_r2_score_test],
                   'MSE train': [format(float(linear_reg_mse_score_train), 'f'), lasso_reg_mse_score_train, ridge_reg_mse_score_train, svr_reg_mse_score_train, decision_tree_reg_mse_score_train, random_forest_reg_mse_score_train],
                   'MSE test': [format(float(linear_reg_mse_score_test), 'f'), lasso_reg_mse_score_test, ridge_reg_mse_score_test, svr_reg_mse_score_test, decision_tree_reg_mse_score_test, random_forest_reg_mse_score_test],
                   'RMSE train': [format(float(linear_reg_rmse_score_train), 'f'), lasso_reg_rmse_score_train, ridge_reg_rmse_score_train, svr_reg_rmse_score_train, decision_tree_reg_rmse_score_train, random_forest_reg_rmse_score_train],
                   'RMSE test': [format(float(linear_reg_rmse_score_test), 'f'), lasso_reg_rmse_score_test, ridge_reg_rmse_score_test, svr_reg_rmse_score_test, decision_tree_reg_rmse_score_test, random_forest_reg_rmse_score_test],
                   'MAE train': [format(float(linear_reg_mae_score_train), 'f'), lasso_reg_mae_score_train, ridge_reg_mae_score_train, svr_reg_mae_score_train, decision_tree_reg_mae_score_train, random_forest_reg_mae_score_train],
                   'MAE test': [format(float(linear_reg_mae_score_test), 'f'), lasso_reg_mae_score_test, ridge_reg_mae_score_test, svr_reg_mae_score_test, decision_tree_reg_mae_score_test, random_forest_reg_mae_score_test],
                   'Explained Variance Score train': [format(float(linear_reg_evs_score_train), 'f'), lasso_reg_evs_score_train, ridge_reg_evs_score_train, svr_reg_evs_score_train, decision_tree_reg_evs_score_train, random_forest_reg_evs_score_train],
                   'Explained Variance Score test': [format(float(linear_reg_evs_score_test), 'f'), lasso_reg_evs_score_test, ridge_reg_evs_score_test, svr_reg_evs_score_test, decision_tree_reg_evs_score_test, random_forest_reg_evs_score_test],
                   'Max Error train': [format(float(linear_reg_me_score_train), 'f'), lasso_reg_me_score_train, ridge_reg_me_score_train, svr_reg_me_score_train, decision_tree_reg_me_score_train, random_forest_reg_me_score_train],
                   'Max Error test': [format(float(linear_reg_me_score_test), 'f'), lasso_reg_me_score_test, ridge_reg_me_score_test, svr_reg_me_score_test, decision_tree_reg_me_score_test, random_forest_reg_me_score_test]
                 }
    resultant_df = pd.DataFrame(data_for_df, index=row_names)

    return resultant_df

model_train_test(x_train=X_train, y_train=Y_train, x_test=X_test, y_test=Y_test)