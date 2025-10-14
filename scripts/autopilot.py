from PyQt6 import QtWidgets

from data_collector import DataCollectionUI

class NNMsgProcessor:
    def __init__(self):
        self.always_forward = True
        # Track current state of controls to avoid sending duplicates
        self.current_state = {
            "forward": False,
            "back": False,
            "left": False,
            "right": False
        }
        
    def nn_infer(self, message):
        #   Do smart NN inference here
        # For now, just go forward
        return {
            "forward": True,
            "back": False,
            "left": False,
            "right": False
        }

    def process_message(self, message, data_collector):
        # Get desired control state from neural network
        desired_state = self.nn_infer(message)
        
        # Only send commands when state changes
        for command, desired in desired_state.items():
            if self.current_state[command] != desired:
                data_collector.onCarControlled(command, desired)
                self.current_state[command] = desired

if  __name__ == "__main__":
    import sys
    def except_hook(cls, exception, traceback):
        sys.__excepthook__(cls, exception, traceback)
    sys.excepthook = except_hook

    app = QtWidgets.QApplication(sys.argv)

    nn_brain = NNMsgProcessor()
    data_window = DataCollectionUI(nn_brain.process_message)
    data_window.show()

    app.exec()