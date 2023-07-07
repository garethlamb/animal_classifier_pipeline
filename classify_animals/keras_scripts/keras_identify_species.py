import os
import json
import logging
import pandas as pd
import numpy as np
import sklearn as skl

from classify_animals.scripts import config
from classify_animals.keras_scripts import preprocess_image
from sklearn import metrics as mt
from sklearn import model_selection
from os import path
from pathlib import Path
from tensorflow import keras as kr


def predict_species(img_dir, model, batch_size = 32):
    """
    Predicts species labels of images
    
    img_dir (PATH or string): Directory contain images
        to be processed
        
    model (Keras model object): Model that will be used to
        make the predictions
    
    Returns Image dataset and the species of the animals
        in the images as predicted by the model
    """
    
    config.block_print() # Block Keras's info messages
    
    # Load data
    dataset = kr.utils.image_dataset_from_directory(
        img_dir,
        labels = None,
        image_size = config.IMAGE_SIZE,
        interpolation = 'bilinear',
        label_mode = 'categorical',
        batch_size = batch_size,
        shuffle = False
    )
    
    config.enable_print() # Re-enable messages to display
    
    # Pre-process data in the same way as how the images were pre-processed
    # When Resnet50 was being trained
    # will convert the images from RGB to BGR, then will zero-center each color channel with
    # respect to the ImageNet dataset without scaling.
    norm_dataset = dataset.map(preprocess_image.main)
    
    y_pred = []
    
    # Loop through all batches of the test data set
    for X in norm_dataset:
        
        batch_pred = model.predict_on_batch(X) # Apply model to test set
        
        y_pred.append(batch_pred) # Add batch's predictions to list of predictions
    
    
    
    # Put predictions into one 2-D array
    y_pred = np.concatenate(y_pred)  
    
    return y_pred, dataset

def entropy(p):
    """
    Calculate the entropy of a discrete probability distribution p.
    (Written by Chat GPT-4)
    
    :param p: probability distribution
    :return: entropy
    """
    return -np.sum(p * np.log(p))

def remove_root_dir(path, root_dir, as_string = True):
    """
    Removes the root dir in a file path.
    (i.e. makes it so that a file path is
    relative to a root directory)
    
    path : file path to be made relative to 
        the root dir
    root_dir : file path to directory that 
        "path" will be made to be relative to
    as_string : If True, output will be a string
    """
    
    p = Path(path) # Turn filepath into a path object
    root = Path(root_dir) # Turn root_dir into a file path
    
    p = p.relative_to(root_dir) # Get realtive file path
    
    if as_string:
        p = str(p)
    
    return p

def keras_identify_species(df_crop, root_dir, img_dir, model, batch_size = 32, ent_thr = None):
    """
    Identifies the species of the animals in all images of a directory
    and adds the species labels to a dataframe that contains the relative
    file paths of all images
    
    df_crop (Dataframe) : must contain a column called 'cropped_image_path'
        that has the relative file path of all images in image dir
    
    root_dir (Path or String): Path to root directory that df_crop's file paths
        are relative to
    
    img_dir (Path or String) : Path to directory that contains images. All
        images in directory will be processed by the classifier
    
    batch_size (int): Number of images to pass through the classifier at a time
    
    ent_thr (float): Threshold value where only predictions with entropies
        below this value will be classified. If None, all images will be classified
    """
    
    if ent_thr == None: # If no threshold was passed
        ent_thr = np.inf # All certainty levels should be accepted
    
    # Get label prediction probabilities
    y_pred_proba, dataset = predict_species(
        img_dir = img_dir, 
        model = model, 
        batch_size = batch_size
    ) 
    
    # Get file paths of images
    file_paths = np.array(dataset.file_paths) 
    
    # Add folder stem to all file paths in image_dir
    file_paths = [remove_root_dir(file_path, root_dir) for file_path in file_paths] 
    
    # Make predictions
    y_pred = np.argmax(y_pred_proba, axis = 1) # Take highest probability as prediction for image
    y_pred = config.SPECIES_CATEGORIES[y_pred] # Find species label associated with category number
    
    # Build dataframe of test data results
    df_results = pd.DataFrame({config.CROP_COL_NAME : file_paths, config.SPECIES_COL_NAME : y_pred})
    
    # Relabel uncertain predictions
    df_results[config.ENT_COL_NAME] = np.apply_along_axis(entropy, axis = 1, arr = y_pred_proba) # Find certainty of predictions
    df_results.loc[ df_results[config.ENT_COL_NAME] > ent_thr, config.SPECIES_COL_NAME] = config.UNKNOWN_SPECIES_NAME # Threshold entropy
    
    # Return df_crop with species labels
    return df_crop.merge(
        df_results,
        on = config.CROP_COL_NAME,
        how = 'left'
    )