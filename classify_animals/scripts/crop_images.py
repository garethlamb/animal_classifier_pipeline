import os
import logging
import pandas as pd
from PIL import Image, ImageOps
from classify_animals.scripts import config
from os import path

def save_crop(img: Image.Image, bb_left_norm, bb_top_norm,
              bb_width_norm, bb_height_norm,
              save: str, square_crop: bool = True) -> bool:
    """
    Crops an image and saves the crop to file. Copied from megadetector's
    example classification project.
    URL: https://github.com/microsoft/CameraTraps/blob/main/classification/crop_detections.py

    Args:
        img: PIL.Image.Image object, already loaded
        bbox_norm: list or tuple of float, [xmin, ymin, width, height] all in
            normalized coordinates
        save: str, path to save cropped image

    Returns: bool, True if a crop was saved, False otherwise
    """
    img_w, img_h = img.size
    xmin = int(bb_left_norm * img_w)
    ymin = int(bb_top_norm * img_h)
    box_w = int(bb_width_norm * img_w)
    box_h = int(bb_height_norm * img_h)
    
    if square_crop:
        # expand box width or height to be square, but limit to img size
        box_size = max(box_w, box_h)
        xmin = max(0, min(
            xmin - int((box_size - box_w) / 2),
            img_w - box_w))
        ymin = max(0, min(
            ymin - int((box_size - box_h) / 2),
            img_h - box_h))
        box_w = min(img_w, box_size)
        box_h = min(img_h, box_size)
    
    if box_w == 0 or box_h == 0:
        logging.info(f'Skipping size-0 crop (w={box_w}, h={box_h}) that would have been saved at {save}')
        
    else:
        
        crop = img.crop(box=[xmin, ymin, xmin + box_w, ymin + box_h])
        
        if square_crop and (box_w != box_h):
            # pad to square using 0s
            crop = ImageOps.pad(crop, size=(box_size, box_size), color=0)
        
        os.makedirs(os.path.dirname(save), exist_ok=True)
        crop.save(save)

def crop_image(sr_bb, source_dir, target_dir, subfolder_name = None):
    """
    Crops an image at a dataframe of bounding boxes and saves
    the crops to a new directory
    
    sr_bb (Dataframe) : Bounding box information for one image.
        Value of path entry must be non-null
    source_dir (string or PATH) : Path to directory that contains
        images that are to be cropped. 'path' variable in sr_bb
        must be a relative path with respect to the source
        directory
    target_dir (string or PATH) : Path to directory where cropped
        image will be stored. Cropped image will be renamed when
        saved to this directory
    
    Returns Dataframe of metainformation on cropped images. Uses
        all of the meta information of the parent image as well as
        the bounding box's confidence level, crop number and path 
        in new directory
    """
    
    # Path to source image relative to source_dir
    relative_source_path = sr_bb['path']
    
    # Full path to source image
    absolute_source_path = path.join(source_dir, relative_source_path)
    
    
    # Try to load image
    try:
    
        with Image.open(absolute_source_path) as img:
     
            # Get file extension of orignal image
            _, ext = os.path.splitext(relative_source_path)
            
            # Get relative file path of file in cropped directory
            relative_target_path = 'crop_' + str(sr_bb.name) + ext
                
            # Add subfolder if it was given
            if subfolder_name != None:
                relative_target_path = path.join(subfolder_name, relative_target_path)
                
            # Get absolute path of cropped image
            absolute_target_path = path.join(target_dir, relative_target_path)
            
            # Crop image and save crop to target path
            save_crop(
                img = img,
                bb_left_norm = sr_bb['bb_left_norm'], 
                bb_top_norm = sr_bb['bb_top_norm'],
                bb_width_norm = sr_bb['bb_width_norm'], 
                bb_height_norm = sr_bb['bb_height_norm'],
                save = absolute_target_path
            )
        
    except:  # If image can't be loaded or cropped
        
        logging.warning(f'Unable to load/crop image at {absolute_source_path}')
        
        relative_target_path = None
        
    return pd.Series({
        config.BASE_COL_NAME : relative_source_path, 
        config.CROP_COL_NAME : relative_target_path,
        config.BB_CONF_COL_NAME : sr_bb[config.BB_CONF_COL_NAME]
    })
        
def main(df_bb, source_dir, target_dir, subfolder_name = None):
    
    
    df_crop = df_bb.apply(
        lambda x : crop_image(
            x,
            source_dir = source_dir,
            target_dir = target_dir,
            subfolder_name = subfolder_name
        ),
        axis = 1
    )
    
    return df_crop