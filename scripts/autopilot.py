from PyQt6 import QtWidgets
import torch
import numpy as np

from data_collector import DataCollectionUI
from neural_network import NeuralNetwork

class NNMsgProcessor:
    def __init__(self):
        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        
        # Load both models
        self.race_model = NeuralNetwork().to(self.device)
        self.safe_model = NeuralNetwork().to(self.device)
        
        self.race_model.load_state_dict(torch.load("scripts/fast_model.pth", map_location=self.device))
        self.safe_model.load_state_dict(torch.load("scripts/slow_model.pth", map_location=self.device))
        
        self.race_model.eval()
        self.safe_model.eval()
        
        # Initialize current_state (this was missing)
        self.current_state = {
            "forward": False,
            "back": False,
            "left": False,
            "right": False
        }
        
    def should_use_safe_mode(self, message):
        # Strategy 1: Distance-based switching
        min_distance = min(message.raycast_distances)
        # return min_distance < 30  # Switch to safe mode if obstacles are within 30 units
        
        # Strategy 2: Speed-based switching
        return message.car_speed > 25 # Switch to safe mode at high speeds
        
        # Strategy 3: Hybrid approach
        # return message.car_speed > 25 or min_distance < 10

    def nn_infer(self, message):
        """
        Use the neural network to predict control commands from sensor data.
        
        message should have:
            - car_speed: float
            - raycast_distances: list of floats (should be 15 values)
        """
        # Extract features from message (same as training: [car_speed, *raycast_distances])
        features = [message.car_speed] + list(message.raycast_distances)
        
        # Convert to numpy array and then to torch tensor
        features_array = np.array([features], dtype=np.float32)  # Add batch dimension
        features_tensor = torch.from_numpy(features_array).to(self.device)
        
        active_model = self.safe_model if self.should_use_safe_mode(message) else self.race_model

        # Run inference
        with torch.no_grad():
            outputs = active_model(features_tensor)  # Shape: [1, 4]
            predictions = (outputs >= 0.5).cpu().numpy()[0]  # Convert to binary and remove batch dim
        

        # Convert predictions to control dictionary
        # Assuming order is: [forward, backward, left, right]
        return {
            "forward": bool(predictions[0]),
            "back": bool(predictions[1]),
            "left": bool(predictions[2]),
            "right": bool(predictions[3])
        }

    def process_message(self, message, data_collector):
        # Get desired control state from neural network
        desired_state = self.nn_infer(message)
        
        # Only send commands when state changes
        for command, desired in desired_state.items():
            if self.current_state[command] != desired:
                print("Changing controls")
                data_collector.onCarControlled(command, desired)
                self.current_state[command] = desired

if  __name__ == "__main__":
    import sys
    def except_hook(cls, exception, traceback):
        sys.__excepthook__(cls, exception, traceback)
    sys.excepthook = except_hook

    app = QtWidgets.QApplication(sys.argv)

    # Load your trained model (update this path to where you save your model)
    nn_brain = NNMsgProcessor()
    data_window = DataCollectionUI(nn_brain.process_message)
    data_window.show()

    app.exec()