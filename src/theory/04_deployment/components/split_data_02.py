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
def split_data(
    in_df: Input[Dataset],
    out_df_train: Output[Dataset], 
    out_df_test: Output[Dataset],
    test_size: float = 0.2,
    random_state: int = 1,
):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_pickle(in_df.path + ".pkl") 
    
    df_train, df_test = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
    )

    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    pd.to_pickle(df_train, out_df_train.path + ".pkl")
    pd.to_pickle(df_test, out_df_test.path + ".pkl")