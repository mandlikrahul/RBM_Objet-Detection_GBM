from RCNN_Object_Detection_From_Scratch.train import Proposed_model


def Proposed_model1(x_train, x_test, y_train, y_test, epochs, percentage):
    Proposed_model(num_epochs=epochs, split=percentage)
