import os

def save_results(results_df, path_to_results_csv):
    os.makedirs(os.path.dirname(path_to_results_csv), exist_ok=True)
    results_df.to_csv(path_to_results_csv)
    print(f'Results has been saved at {path_to_results_csv}!')
