from sklearn.model_selection import cross_validate

def train_model(model, x, y):
    """
    Trains a model, applies cross-validation and gets the results

    Returns:
        - Model
        - Results dictionary
    """
    # We apply cross-validation to check the model's general performance during training
    cv_performances = cross_validate(model, x, y, scoring=['accuracy', 'recall', 'precision'])

    avg_accuracy = cv_performances['test_accuracy'].mean()
    avg_recall = cv_performances['test_recall'].mean()
    avg_precision = cv_performances['test_precision'].mean()

    # Now we know how good it does in general, let's train it on all the training data
    model.fit(x, y)

    print(f"Our average cross-validation accuracy for {type(model).__name__} is: \n"
          f"\t - Score for accuracy is {avg_accuracy}. \n"
          f"\t - Score for recall is {avg_recall}. \n"
          f"\t - Score for precision is {avg_precision} \n"
    )

    results = {
        'model_name': type(model).__name__,
        'cv_accuracy': avg_accuracy,
        'cv_precision': avg_precision,
        'cv_recall': avg_recall
    }

    return model, results
