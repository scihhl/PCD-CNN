from sklearn.ensemble import RandomForestClassifier


class RandomForestModel:
    def __init__(self, features, target, n_estimators=100, random_state=42):
        """
        Initialize a RandomForestModel with specified parameters.
        :param features: List of feature column names used for training the model.
        :param target: Target column name that the model should predict.
        :param n_estimators: Number of trees in the forest.
        :param random_state: Seed used by the random number generator for reproducibility.
        """
        self.features = features
        self.target = target
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = RandomForestClassifier(n_estimators=self.n_estimators, random_state=self.random_state)
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    def input_data(self, train_data, test_data):
        """
        Input training and testing data into the model.
        :param train_data: DataFrame containing the training data.
        :param test_data: DataFrame containing the testing data.
        """
        self.X_train = train_data[self.features]
        self.y_train = train_data[self.target]
        self.X_test = test_data[self.features]
        self.y_test = test_data[self.target]

    def train(self):
        """
        Train the RandomForest model using the training data.
        """
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        """
        Evaluate the trained model using the testing data and print the accuracy.
        """
        accuracy = self.model.score(self.X_test, self.y_test)
        print(f"Accuracy: {accuracy}")

    def predict(self, X):
        """
        Make predictions using the model.
        :param X: DataFrame containing the data for making predictions.
        :return: Array of predictions.
        """
        X = X[self.features]
        return self.model.predict(X)


