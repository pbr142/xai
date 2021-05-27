import numpy as np
import pandas as pd
from category_encoders.hashing import HashingEncoder
from optuna.integration import OptunaSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def default_pipeline(X: pd.DataFrame, high_cardinality_threshold: int=11, numeric_types=[int, float], categorical_types='object') -> ColumnTransformer:
    """Create a pipeline to process a DataFrame for a scikit-learn model.
    
    * For numeric features, standardization is applied.
    * For low cardinality categorical features, one-hot encoding is applied.
    * For high cardinality categorical features, hashing encoding is applied.

    Args:
        X (pd.DataFrame): DataFrame with features
        high_cardinality_threshold (int, optional): Thresholds for number of categories to distinguish high cardinality and low cardinality categorical features. Defaults to 11.
        numeric_types (list, optional): Types to identify numeric features. Defaults to [int, float].
        categorical_types (str, optional): Types to identify categorical features. Defaults to 'object'.

    Returns:
        ColumnTransformer: [description]
    """

    # define columns
    numeric_columns = X.select_dtypes(numeric_types).columns
    categorical_columns = X.select_dtypes(categorical_types).columns
    idx_high_cardinality = np.array([len(X[col].unique()) >= high_cardinality_threshold for col in categorical_columns])
    high_cardinality_columns = categorical_columns[idx_high_cardinality]
    low_cardinality_columns = categorical_columns[~idx_high_cardinality]

    # define pipelines
    numeric_pipeline = make_pipeline(StandardScaler())
    low_cardinality_pipeline = make_pipeline(OneHotEncoder(handle_unknown='ignore'))
    high_cardinality_pipeline = make_pipeline(HashingEncoder(return_df=False))
    feature_pipeline = ColumnTransformer([
        ('numeric', numeric_pipeline, numeric_columns),
        ('low_cardinality', low_cardinality_pipeline, low_cardinality_columns),
        ('high_cardinality', high_cardinality_pipeline, high_cardinality_columns)
    ], remainder='passthrough')

    return feature_pipeline


def fit_model(model, param_distributions: dict, X: pd.DataFrame, y: pd.Series):
    feature_pipeline = default_pipeline(X)
    model = Pipeline([
        ('features', feature_pipeline),
        ('model', model)
    ])

    param_distributions = {'model__'+k:v for k,v in param_distributions.items()}

    model_search = OptunaSearchCV(model, param_distributions, n_trials=20*len(param_distributions),
    n_jobs=-1, random_state=142, verbose=0)

    model_search.fit(X, y)
    best_model = model_search.best_estimator_
    setattr(best_model, 'training_data', (X, y))

    return best_model