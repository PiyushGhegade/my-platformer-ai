import tensorflow as tf
from tensorflow import keras
import numpy as np

class AIModel:
    def __init__(self, state_size, action_size):
        self.state_size = state_size  # Number of input features (e.g., player_x, enemy_x, etc.)
        self.action_size = action_size  # Number of possible actions (e.g., left, right, jump)
        self.model = self._build_model()

    def _build_model(self):
        """Creates a Deep Q-Network model"""
        model = keras.Sequential([
            keras.layers.Dense(64, activation="relu", input_shape=(self.state_size,)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(self.action_size, activation="softmax")  # Probabilities for each action
        ])
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        return model

    def predict(self, state):
        """Predicts the best action for the given game state"""
        state = np.array(state).reshape(1, -1)  # Convert state to a batch format
        action_probs = self.model.predict(state, verbose=0)
        return np.argmax(action_probs)  # Returns the action with the highest probability

    def train(self, X_train, y_train, epochs=10):
        """Trains the AI model on collected game data"""
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split=0.1)

    def save_model(self, filename="mario_ai_model.h5"):
        """Saves the trained model"""
        self.model.save(filename)

    def load_model(self, filename="mario_ai_model.h5"):
        """Loads a pre-trained model"""
        self.model = keras.models.load_model(filename)
