def train_model(model, X_train: np.array, Y_train: np.array, X_test: np.array):
    if not all(isinstance(arr, np.ndarray) for arr in [X_train, Y_train, X_test]):
        raise TypeError('X_train, Y_train, X_test should be arrays')

    print(f'Initializing...: {model}')
    
    # Assuming get_models() is a function that returns a dictionary of models
    models_dict = get_models()

    if model not in models_dict:
        raise ValueError(f'Model {model} not found in the available models.')

    print('Training....')
    selected_model = models_dict[model]
    trained_model = selected_model.fit(X_train, Y_train)

    print('Testing.....')
    y_pred = trained_model.predict(X_test)

    return y_pred