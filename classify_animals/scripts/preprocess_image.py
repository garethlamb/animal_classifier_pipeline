from tensorflow import keras as kr

def main(image_data):
    """
    Pre-processes image data so that it can be fed
    into the trained neural network
    
    image_data (Tensorflow image) : Image to be
        pre-processed
    """

    return kr.applications.resnet50.preprocess_input(image_data)