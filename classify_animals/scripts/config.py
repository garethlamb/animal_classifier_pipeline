import os
import sys
import json
import shutil
import json
import pickle
import pandas as pd
import numpy as np
from os import path

"""----------------- Global Variables ---------------------"""

# Size of NN input images
IMAGE_SIZE = (224, 224)

# Species for species above entropy threshold
UNKNOWN_SPECIES_NAME = 'zz_unknown' 

# Real-world meaning of numerical labels outputted
# by the neural network
SPECIES_CATEGORIES = np.array([
    'Bird', 
    'Canis lupus familiaris', 
    'Eurasian Otter', 
    'Felis catus',
    'Herpestes javanicus',
    'Hystrix brachyura',
    'Macaca mulatta',
    'Melogale species',
    'Muntiacus species',
    'Other animal',
    'Paguma larvata',
    'Prionailurus bengalensis',
    'Rodent',
    'Sus scrofa',
    'Viverricula indica'
])

# Default name of working directory
WORK_DIR_DEFAULT_NAME = 'Animal Detections'

# Column names of CSV that describe the animal
# detected to be within the images
BASE_COL_NAME = 'original_image_path'
CROP_COL_NAME = 'cropped_image_path'
SPECIES_COL_NAME = 'species'
BB_CONF_COL_NAME = 'bb_conf'
ENT_COL_NAME = 'class_ent'

# File names of PyTorch model
CLASSIFIER_NAME = 'classifier.onnx'
EMBEDDING_MODEL_NAME = 'embedding_model_weights.pt'
EMBEDDING_MEAN_NAME = 'dataset_mean.npy'
EMBEDDING_STD_NAME = 'dataset_std.npy'

def build_working_file_dict(working_dir):
    """
    Build dictionary containing all absoulute file paths
    and directories used by the program
    """
    
    work_dict = {}
    
    # Define Directories
    work_dict['unsorted_cropped_images_dir'] = path.join(working_dir, 'unsorted_cropped_images')
    work_dict['sorted_cropped_images_dir'] = path.join(working_dir, 'cropped_images')
    work_dict['checkpoint_dir'] = path.join(working_dir, 'checkpoints')
    
    # Define filepaths
    work_dict['bb_csv_path'] = path.join(work_dict['checkpoint_dir'], 'bounding_boxes.csv')
    work_dict['crop_checkpoint_path'] = path.join(work_dict['checkpoint_dir'], 'labels_checkpoint.pkl')
    work_dict['batch_number_path'] = path.join(work_dict['checkpoint_dir'], 'batch_record.json')
    work_dict['output_path'] = path.join(working_dir, 'Animal Labels.csv')
    
    #work_dict['trained_weights_path'] = path.join(working_dir, 'data', 'trained_model.h5')
    
    
    return work_dict

def save_csv(df, directory, filename = None, save_index = True):
    """Writes a csv to a directory and creates that directory
        if it doesn't already exist"""
    
    # If a filename was not passed to the function
    if filename == None: 
        
        # Assume that file name is at the end of the path
        dir_tokens = os.path.split(directory)
        directory = dir_tokens[0]
        filename = dir_tokens[1]
    
    # If output directory doesn't exist
    if not os.path.exists(directory): 
        os.makedirs(directory) # Make output directory
    
    df.to_csv(directory + '\\' + filename,
              index = save_index)
              
def move_to_new_dir(old_path, new_path):
    """
    Sends file from old path to new path and
    creates the directory fo the new path if it doesn't
    already exist
    """
    
    # Assume that file name is at the end of the path
    dir_tokens = os.path.split(new_path)
    new_directory = dir_tokens[0]
    
    # If output directory doesn't exist
    if not os.path.exists(new_directory): 
        os.makedirs(new_directory) # Make output directory
    
    
    # Rename file and move it to new directory
    shutil.move(old_path, new_path)
              

def read_bounding_boxes(bounding_boxes_path):
    """
    Loads the bounding boxes that were produced by 
    megadetector
    
    Returns data frame that has one row for each
    bounding box and the columns are the qualities
    of the bounding box. Also returns info on how
    the bounding boxes were made
    """
    
    # Load json string that contains bounding box
    # data
    with open(bounding_boxes_path, 'r') as f:
        js = json.load(f)
    
    # Empty array of records to store bounding box data
    bb_list = [] 
    
    for img in js['images']: # For each image
        
        if 'detections' in img: # Image has a bounding box
            
            for detection in img['detections']: # For each bounding box in image
            
                bb_list.append({
                    'path' : img['file'], # Relative path to image
                    'category' : detection['category'], # Type of detection
                    BB_CONF_COL_NAME : detection['conf'], # Confidence of detection
                    'bb_left_norm' : detection['bbox'][0], # Normalised left-most coordinate
                    'bb_top_norm' : detection['bbox'][1], # Normalised upper-most coordinate
                    'bb_width_norm' : detection['bbox'][2], # Normalised width of bb
                    'bb_height_norm' : detection['bbox'][3], # Normalised height of bb
                })
        else:
            bb_list.append({
                'path' : img['file'], # Relative path to image
                'category' : img['failure'], # Type of detection
            })
    
    # Get names of categories
    detection_categories = js['detection_categories']
    
    # Build dataframe of bounding boxes
    df_bb = pd.DataFrame.from_records(bb_list)
    
    # Give detection categories more meaningful names
    df_bb['category'] = df_bb['category'].replace(detection_categories)
    
    # Return bounding boxes and information about how they were made
    return df_bb, js['info']

  
def can_file_be_edited(f, error_message = None):
    """
    Copied from https://stackoverflow.com/questions/11114492/check-if-a-file-is-not-open-nor-being-used-by-another-process
    
    Checks if a file is already open by another process and raises an
    error if it is. (This only works on Windows)
    
    f (String): Absolute file path to file
    """
    
    if error_message == None: # If no Error message has been specified
    
        # Set error message to default
        error_message = 'Could not edit file ' + f

    if os.path.isfile(f):
        try:
            # Try to make a trivial edit to the file
            os.rename(f, f)
        
        except:
            raise Exception(error_message)
            
def sort_by_species(df, old_dir, new_dir, species, path_col_name):
    """
    Moves/copies files for one species to new directory 
    that has a file structure that can be read by Keras
    
    df (Dataframe) : Dataframe object containing filenames of images
        to be sent to new directory. All images must have the same
        species label
    old_dir (String) : Root directory where cropped images are stored
        before being sent to new directory
    new_dir (String) : Root directory that images will be sent to. Images
        will be placed in a subfolder of this directory that has the species
        name as its title
    species (string) : Name of species for all images in data frames
    path_col_name (string) : Name of column that contains relative file
        paths to images in old_dir
        
    Returns dataframe with the image's new file paths that are relative
        to the new root directory
    """
    
    # Ensure dataframe is passed by value
    df = df.copy()
    
    # Get name of folder that images will be placed in
    folder_name = species
    
    
    # For each crop in the class of species
    for i in range(df.shape[0]):
        
        sr_crop = df.iloc[i] # Get data on crop
        
        
        # Get old absolute file path
        old_full_path  = old_dir + '\\' + sr_crop[path_col_name]
        
        # Get old file extension
        file_ext = os.path.splitext(sr_crop[path_col_name])[1]
        
        # Get file's new path details
        new_filename = species + '_image_' + str(i + 1)
        new_relative_path = folder_name + '\\' + new_filename + file_ext
        new_full_path = new_dir + '\\' + new_relative_path 
        
        # Send image to its new file
        move_to_new_dir(old_full_path, new_full_path)
        
        # Replace image's path to its new location
        df[path_col_name].iloc[i] = new_relative_path
        
    return df
    
def delete_empty_folders(root_folder):
    """
    Recursively deletes all empty folders in a directory.
    
    Written by Chat GPT 3.5
    """
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)
        if os.path.isdir(folder_path):
            delete_empty_folders(folder_path)
            if not os.listdir(folder_path):
                os.rmdir(folder_path)

def delete_folder_if_empty(root_folder):
    """
    Deletes all empty folders in the directory and
    the directory itself if it becomes empty after
    deleting all of the empty folders
    """
    
    # Remove all empty folders in directory
    delete_empty_folders(root_folder)
    
    # If after deleting all empty folders, the
    # root directory is now empty
    if not os.listdir(root_folder):
    
        # Delete the now empty directory
        os.rmdir(root_folder)

def block_print():
    """
    Disable print messages.
    
    Copied from https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print
    """
    
    sys.stdout = open(os.devnull, 'w')

def enable_print():
    """
    Enable print messages.
    
    Copied from https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print
    """
    
    sys.stdout = sys.__stdout__
    
def all_checkpoints_exist(path_dict):
    """
    Checks if all files needed for a checkpoint exist
    in checkpoint folder
    
    path_dict (dict) : Dictionary of file paths of files
        that are used by the program
    
    Returns True if all checkpoint files exist and False
        otherwise
    """
    
    # Check if bounding box information exists
    if not os.path.exists(path_dict['bb_csv_path']):
    
        return False
    
    # Check if information on existing crops exists
    if not os.path.exists(path_dict['crop_checkpoint_path']):
        
        return False
    
    # Check if marker for where in bounding box list the
    # program should resume from exists
    if not os.path.exists(path_dict['batch_number_path']):
        
        return False
    
    # Return True if all files needed for a checkpoint exists
    return True

def initialise_checkpoints(bb_results_path, md_thr):
    """
    Initialises datastructures used to loop
    through the data and a dataframe of bounding
    boxes that the program will later loop through
    
    bb_results_path (String or Python PATH object): Path to 
        Megadetector's JSON string output for bounding boxes. 
        Can either by a Pyrhon PATH object, absolute or 
        relative path
    
    md_thr (float) : Threshold value for the confidence ratings of
        Megadetector's detections. Only bounding boxes with a confidence
        rating above the threshold will be cropped and analysed. A higher
        threshold means that detections are more likely to be of an animal
        but it also makes it more likely that fewer animals will be detected
    
    Returns bounding box information, list to store information
    on cropped data and the start row number in that order
    """
    
    # Read bounding boxes
    df_bb, _ = read_bounding_boxes(bb_results_path)
    
    # Get only animal detections above confidence threshold
    df_bb = df_bb[
        (df_bb['category'] == 'animal') &
        (df_bb[BB_CONF_COL_NAME] >= md_thr)
    ]
    
    # Reset row numbers
    df_bb = df_bb.reset_index(drop = True) 
    
    # Return bounding box information, list to store information
    # on cropped data and the start row number in that order
    return df_bb, [], 0

def save_checkpoint(crops, start_row_number, path_dict):
    """
    Saves checkpoints of information on crops and of which
    part of the data the program should resume classifying
    from
    
    crops (list of data frames): Information on animal crops
        from all processed batches
    start_row_number (int): Row number of data frame of
        bounding box information that the program should resume
        from when it loads a checkpoint
    path_dict (dict) : Dictionary of file paths of files
        that are used by the program
    
    """
    
    # Save species labels
    with open(path_dict['crop_checkpoint_path'], 'wb') as file:
        pickle.dump(crops, file)
        
    # Save batch number
    with open(path_dict['batch_number_path'], 'w') as file:
        json.dump(start_row_number, file)