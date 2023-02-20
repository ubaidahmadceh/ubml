.. _regression:

Regression
============

Import Statement::

    from ubml.train_test import regression_train_test


.. function:: regression_train_test(x_train, y_train, x_test, y_test, metric="r2_score", export_best=None, export_model=None, path=None)

   :param x_train: the x_train
   :type x_train: output from sklearn train_test_split()

   :param y_train: the y_train
   :type y_train: output from sklearn train_test_split() 

   :param x_test: the x_test
   :type x_test: output from sklearn train_test_split()

   :param y_test: the y_test
   :type y_test: output from sklearn train_test_split()

   :param metric: Give it a metric name for which you want to choose best model. By default it will take value as r2_score and choose the best model with the highest validation r2_score, but you can change the metric
   :type metric: str

   :param export_best: set it as True if you want to export the best model as a pickle file
   :type export_best: bool

   :param export_model: give it a model name if you want to export that specific model's pickle file
                            models can be selected as follows:
                                * Linear Regression.
                                * Lasso Regression.
                                * Ridge Regression.
                                * Support Vector Regression.
                                * Decision Tree Regression.
                                * Random Forest Regression.
   :type export_model: str

   :param path: give a path with model name if you want to specify where to save the model and by what name
   :type path: str

   :returns: * Metrics Performance Table (type: A Pandas Dataframe).
             * Best Model Name.
             * Best Model pickle file (only if you set argument export_best=True).
             * Custom Model pickle file (only if you set argument export_model=True).   