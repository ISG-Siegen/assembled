def _default_preprocessor():
    from sklearn.compose import ColumnTransformer
    from sklearn.compose import make_column_selector
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="constant", fill_value=-1),
             make_column_selector(dtype_exclude="category")),
            ("cat", Pipeline(steps=[("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                                    ("imputer", SimpleImputer(strategy="constant", fill_value=-1))
                                    ]),
             make_column_selector(dtype_include="category")),
        ],
        sparse_threshold=0
    )

    return preprocessor


def check_fold_data_for_ensemble(X_train, X_test, y_train, y_test, preprocessor):
    """Check data for ensemble evaluation

    Apply (default) preprocessing to X_train and X_tests and guarantees that every returned value is a numpy array.
    """
    if preprocessor is None:
        preprocessor = _default_preprocessor()

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    # Make sure y_train and y_test are arrays
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    return X_train, X_test, y_train, y_test
