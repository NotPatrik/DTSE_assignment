import joblib


class Model:
    def __init__(self, model):
        self.model = joblib.load(model)

    def train_model(self, x_train, y_train):
        self.model.fit(x_train, y_train)
        print("Model trained successfully.")

    def predict(self, data):
        predictions = self.model.predict(data)
        return predictions