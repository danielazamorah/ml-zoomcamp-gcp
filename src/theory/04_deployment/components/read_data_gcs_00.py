# Imports
from kfp.dsl import (
    Output, 
    Dataset,
    component,
)

# Components
@component(
    base_image="gcr.io/deeplearning-platform-release/base-cpu.py310:latest",
    packages_to_install=[
        "gcp_python_client_functions",
    ],
)
def read_data_gcs(
    project_id: str, 
    data_gcs_uri: str,
    out_df: Output[Dataset], 
):
    import pandas as pd
    import pickle
    
    from gcp_python_client_functions.clients import Storage
    ##################### CODE:
    stg_obj = Storage(project_id)

    df = pd.read_csv(data_gcs_uri)

    # Base formatting:
    df.columns = df.columns.str.lower()
    categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)
    for c in categorical_columns:
        df[c] = df[c].str.lower()

    pd.to_pickle(df, out_df.path + ".pkl")