import os
import itertools

def save_results(results_df, path_to_results_csv):
    os.makedirs(os.path.dirname(path_to_results_csv), exist_ok=True)
    results_df.to_csv(path_to_results_csv)
    print(f'Results has been saved at {path_to_results_csv}!')

def all_column_combinations(df):
    nested_combinations = list(
        itertools.chain(
            [list(
                itertools.combinations(df.columns, i)) for i in range(2, len(df.columns))
            ]
        )
    )

    flattened_combinations = [item for i in nested_combinations for item in i]

    return flattened_combinations
