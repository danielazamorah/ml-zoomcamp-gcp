# Imports
from kfp.dsl import (
    Input, 
    Output, 
    ClassificationMetrics,
    Model,
    Dataset, 
    component,
)
@component(
    base_image="gcr.io/deeplearning-platform-release/base-cpu.py310:latest",
    packages_to_install = ["scikit-learn==1.3.1"],
)
def evaluate(
    in_df_test: Input[Dataset], 
    cat_features: list,
    num_features: list,
    label: str,
    in_model_pipeline: Input[Model], 
    out_metrics: Output[ClassificationMetrics],
):

    import pandas as pd
    import joblib
    from sklearn.metrics import roc_auc_score

    df_test = pd.read_pickle(in_df_test.path + ".pkl") 
    model_pipeline = joblib.load(in_model_pipeline.path + '.joblib')
    
    # Make prediction:
    y_test = df_test[label].copy()
    x_test = df_test[cat_features + num_features].copy()

    y_pred = model_pipeline.predict_proba(x_test)[:,1]
    
    # Evaluate:
    auc = roc_auc_score(y_test, y_pred)

    out_metrics.metadata = {"auc":auc}
    
    print(f'auc={auc}')