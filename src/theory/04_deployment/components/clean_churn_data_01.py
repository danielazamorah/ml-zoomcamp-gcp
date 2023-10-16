# Imports
from kfp.dsl import (
    Input, 
    Output, 
    Dataset,
    component,
)
@component(
    base_image="gcr.io/deeplearning-platform-release/base-cpu.py310:latest",
)
def clean_churn_data(
    in_dataframe: Input[Dataset],
    out_df_prepared: Output[Dataset], 
):
    import pandas as pd

    dataframe = pd.read_pickle(in_dataframe.path + ".pkl")  

    df_prepared = dataframe.copy()
    df_prepared.totalcharges = pd.to_numeric(df_prepared.totalcharges, errors='coerce') # coerse: bad non numeric values to NaN
    
    # Handle null values
    df_prepared.totalcharges = df_prepared.totalcharges.fillna(0)

    # Binary label (we'll handle the rest of the variables later):
    df_prepared.churn = (df_prepared.churn == 'yes').astype(int)

    pd.to_pickle(df_prepared, out_df_prepared.path + ".pkl")