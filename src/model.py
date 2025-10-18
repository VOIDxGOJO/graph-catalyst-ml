from sklearn.neural_network import MLPClassifier, MLPRegressor

def build_models(hidden_layers=(128, 64), random_state=42):

    clf = MLPClassifier(hidden_layer_sizes=hidden_layers, max_iter=500, random_state=random_state)
    reg = MLPRegressor(hidden_layer_sizes=hidden_layers, max_iter=500, random_state=random_state)
    return clf, reg
