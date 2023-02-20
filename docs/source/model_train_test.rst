.. _model_train_test:

model_train_test
============

Import Statement::

    from ubml.train_test import model_train_test


.. function:: model_train_test(mode, x_train, y_train, x_test, y_test)

    :param mode: set it as "regression", if you want to do regression OR set it as "classification", if you want to do classification
    :type mode: str

    :param x_train: the x_train
    :type x_train: output from sklearn train_test_split()

    :param y_train: the y_train
    :type y_train: output from sklearn train_test_split() 

    :param x_test: the x_test
    :type x_test: output from sklearn train_test_split()

    :param y_test: the y_test
    :type y_test: output from sklearn train_test_split()

    :returns: * Metrics Performance Table (type: A Pandas Dataframe).
              * Best Model Name.
              * Best Model pickle file (only if you set argument export_best=True).
              * Custom Model pickle file (only if you set argument export_model=True).