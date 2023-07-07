import logging
import torch
import pickle
from tqdm import tqdm
import numpy as np
from classify_animals.scripts import config
from classify_animals.pytorch_scripts.networks import NormalizedEmbeddingNet, SoftmaxNet
from onnxruntime import InferenceSession
from sklearn.preprocessing import MinMaxScaler

def load_pytorch_model(embedding_model_path, classifier_path, device):
    """
    Loads pytorch embedding model and onnx classifier
    
    embedding_model_path (path or string) : Path to embedding
        model weights and other parameters. Expects that model
        weights were saved using the file format in:
    
    classifier_path (path or string) : Path to pickled trained 
        Sci-Kit learn classifier
        
    device (torch.device or string): Device to load the model
        on (usually either 'cpu' for CPU or 'cuda' for GPU)
    """

    # Load embedding model and its training objects
    embedding_model_settings = torch.load(
        f = embedding_model_path,
        map_location = device
    )
    
    # Instantiate embedding model
    if embedding_model_settings['loss_type'].lower() == 'softmax':
        embedding_model = SoftmaxNet(
            architecture = embedding_model_settings['arch_name'], 
            feat_dim = embedding_model_settings['feat_dim'], 
            num_classes = len(config.SPECIES_CATEGORIES), 
            use_pretrained = False
        )
    else:
        embedding_model = NormalizedEmbeddingNet(
            architecture = embedding_model_settings['arch_name'], 
            feat_dim = embedding_model_settings['feat_dim'], 
            use_pretrained = False
        )
    # Set up embedding model with parallel processing
    embedding_model = torch.nn.DataParallel(embedding_model)
    
    # Assign saved weights to embedding model
    embedding_model.load_state_dict(embedding_model_settings['embedding_model_state_dict'])
    
    # Find device for running classifier
    provider = 'CPUExecutionProvider' if device == 'cpu' else 'CUDAExecutionProvider'
    
    # Load classifier
    sess = InferenceSession(classifier_path, providers = [provider])
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    
    # Assemble classifier parameters that are needed
    # to make predictions
    classifier = {
        'sess' : sess,
        'input_name' : input_name
    }
    
    return embedding_model, classifier, embedding_model_settings['feat_dim']

class Engine():
    """
    Class for loading feature extractor and classifier. Also
    provides methods for applying these models on a given
    data loader. Adapted from
    https://github.com/microsoft/CameraTraps/blob/main/research/active_learning/deep_learning/engine.py
    """
    

    def __init__(self, embedding_model, classifier, feat_dim, device):
        
        # Attributes
        self.embedding_model = embedding_model.to(device)
        self.device = device
        self.feat_dim = feat_dim
        
        # Unpack classifier
        self.sess = classifier['sess']
        self.input_name = classifier['input_name']
        
    def classify(self, embedding):
        """
        Classify images from their embedding
        
        embedding (Numpy 2D array) : Embedding of chosen
            images where each row is from a different
            image and the columns are the extracted
            features
        """
        
        # Get labels and prediction probabilities
        pred = self.sess.run(None, {self.input_name : embedding})
        
        # Return predicted labels only
        return pred[0]

    def predict(self, dataloader, normalise = True, show_progress = True):
        """
        Predict labels for all data in dataloader. Data loader
        should not be shuffled as to preserve order of indices.
        This method simply performs the embedding and classify methods
        in sequence
        
        Returns 1D numpy array of numerical encodings of labels of images
            given by the dataloader in the same order as they were given
            by the loader
        """
        
        return self.classify(
            self.embedding(
                dataloader = dataloader,
                normalise = normalise,
                show_progress = show_progress
            )
        )

    def embedding_one_batch(self, input):
        """
        Helper function for embedding method. Extracts
        embedding for one batch
        """
    
        with torch.no_grad():
            input = input.to(self.device)
            # compute output
            _, output = self.embedding_model(input)

        return output

    def embedding(self, dataloader, normalise = True, show_progress = True):
        """
        Extracts features from all images given by the
        dataloader. The "embedding" in this case refers
        to the features that have been extracted
        """
        
        # switch to evaluate mode
        self.embedding_model.eval()
        
        embeddings = np.zeros((len(dataloader.dataset), self.feat_dim), dtype=np.float32)
        k = 0
        
        for i, batch in enumerate(tqdm(dataloader, desc = "Embedding:", disable = not show_progress)):
            
            images=batch[0]
            embedding= self.embedding_one_batch(images)
            embeddings[k:k+len(images)] = embedding.data.cpu().numpy()
            k += len(images)
        
        if normalise:
            scaler = MinMaxScaler()
            return scaler.fit_transform(embeddings)
        else:
            return embeddings