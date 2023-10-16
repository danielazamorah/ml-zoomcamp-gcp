# Imports
from kfp.dsl import (
    Input, 
    Output, 
    Dataset,
    Model, 
    ClassificationMetrics,
    component,
)
@component(
    base_image="gcr.io/deeplearning-platform-release/base-cpu.py310:latest",
    packages_to_install = ["scikit-learn==1.3.1"],
)
def logreg_prep_train(
    in_df_train: Input[Dataset],
    cat_features: list,
    num_features: list,
    label: str,
    out_model_pipeline: Output[Model],
    out_metrics: Output[ClassificationMetrics],
    kfold_splits: int = 1, # Validation
    C: int = 1, # Hyperparameter
):
    import pandas as pd 
    import numpy as np

    import joblib
    import pickle

    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import make_column_transformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline

    df_train = pd.read_pickle(in_df_train.path + ".pkl") 

    y_train = df_train[label].copy()
    x_train = df_train[cat_features + num_features].copy()

    def train(x_train, y_train, cat_features):
        ohe = OneHotEncoder(
                    drop='first', # Whether to drop one of the features
                    sparse=False, # Will return sparse matrix if set True
                    handle_unknown='error' # Whether to raise an error 
                ) 
        column_transform = make_column_transformer(
                    (ohe, cat_features),
                    remainder='passthrough',
                )
        regr = LogisticRegression(C=C, max_iter=1000)
        model_pipeline = make_pipeline(column_transform, regr)
        model_pipeline.fit(x_train, y_train)

        return model_pipeline
    
    # Validation:
    kfold = KFold(n_splits=kfold_splits, shuffle=True, random_state=1)
    k_folds_scores = []
    fold = 0

    # Validate:
    for train_idx, val_idx in kfold.split(x_train):
        x = x_train.iloc[train_idx].copy()
        x_val = df_train.iloc[val_idx].copy()
        
        y = y_train.iloc[train_idx]
        y_val = y_train.iloc[val_idx]

        model_pipeline = train(x, y, cat_features)

        y_pred = model_pipeline.predict_proba(x_val)[:,1]

        auc = roc_auc_score(y_val, y_pred)
        k_folds_scores.append(auc)
        
        print(f'auc on fold {fold} is {auc}')
        fold += 1

    mean_auc = np.mean(k_folds_scores)
    std_auc = np.std(k_folds_scores)

    out_metrics.metadata = {
        "mean_auc":mean_auc,
        "std_auc":std_auc,
    }

    print('validation results:')
    print('C=%s %.3f +- %.3f' % (C, mean_auc, std_auc))

    # Train:
    model_pipeline = train(x_train, y_train, cat_features)

    joblib.dump(out_model_pipeline, out_model_pipeline.path + '.joblib')

    with open(out_model_pipeline.path+'.joblib', 'wb') as f:
        pickle.dump(model_pipeline, f)