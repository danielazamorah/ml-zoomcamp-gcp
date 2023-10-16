# Imports
from kfp import dsl

from components.read_data_gcs_00 import read_data_gcs
from components.clean_churn_data_01 import clean_churn_data
from components.split_data_02 import split_data
from components.logreg_prep_train_03 import logreg_prep_train
from components.evaluate_04 import evaluate

# Pipeline
@dsl.pipeline(
    name="pipeline-log-reg",
)
def pipeline(
    project_id: str, 
    data_gcs_uri: str,
    cat_features: list,
    num_features: list,
    label: str,
):    
    o1 = read_data_gcs(
        project_id = project_id, 
        data_gcs_uri = data_gcs_uri,
    )
    o2 = clean_churn_data(
        in_dataframe = o1.outputs["out_df"],
    )
    o3 = split_data(
        in_df = o2.outputs["out_df_prepared"],
    )
    o4 = logreg_prep_train(
        in_df_train = o3.outputs["out_df_train"],
        cat_features = cat_features,
        num_features = num_features,
        label = label,
        kfold_splits = 5,
    )
    o5 = evaluate(
        in_df_test = o3.outputs["out_df_test"], 
        cat_features = cat_features,
        num_features = num_features,
        label = label,
        in_model_pipeline = o4.outputs["out_model_pipeline"], 
    )