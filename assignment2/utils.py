def remove_null_features(df, threshold = 0.5):
    print(f"\tWe will remove null values larger than {threshold}")
    for column in df.columns:
        if df[column].isnull().sum()/len(df) > threshold:
            print(f"Column {column} will be dropped, has {df[column].isnull().sum()} nulls.")
            df = df.drop(columns=column, axis=1)

    return df
