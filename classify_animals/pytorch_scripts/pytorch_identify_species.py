from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from os.path import join as jpth
from classify_animals.scripts import config
from torchvision.transforms import CenterCrop, Normalize, Resize, Compose, ToTensor


class CSVImageDataset(Dataset):
    """
    Creates a PyTorch dataset from a CSV of filepaths
    
    path_col : Column name of CSV that contains
        relative path to images
    
    Adapted from https://pytorch.org/tutorials/beginner/basics/data_tutorial.html?highlight=dataset
    """

    def __init__(self, data, img_dir, path_col, classes, label_col = 'label', 
        targets_col = 'label_num', transform = None, target_transform = None, ):
        """
        Sets up PyTorch dataset from CSV of file paths.
        """
        
        # Attributes
        self.df_data = data.copy()
        self.img_dir = img_dir
        self.target_transform = target_transform
        self.transform = transform
        self.path_col = path_col
        self.label_col = label_col
        self.targets_col = targets_col
        self.classes = classes
        self.classes.sort()
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Validate dataframe
        columns = self.df_data.columns
        assert self.path_col in columns
        if not(self.label_col in columns): self.df_data[self.label_col] = None
        if not(self.targets_col in columns): self.df_data[self.targets_col] = np.NINF
        
    def __len__(self):
        return len(self.df_data)

    def __getitem__(self, idx):
        """
        Returns image data and label
        of the image at the given index
        """
    
        # Get image
        img_path = jpth(self.img_dir, self.df_data.loc[idx, self.path_col])
        image = Image.open(img_path)
        if self.transform is not None:
            image = self.transform(image)
        
        
        # Get label
        label = self.df_data.loc[idx, self.targets_col]
        if self.target_transform is not None and not np.isinf(label):
            label = self.target_transform(label)
            
        # Return image label pair
        return image, label
    
    def get_data_loader(self, batch_size):
        return DataLoader(self, batch_size = batch_size)
    
    def update_labels(self, label_nums):
        """
        Assigns labels to data. Updates both
        numerical encoding of labels and labels
        in word form
        
        label_nums (dict) : New label numbers.
            The key is the image's index in the
            dataframe and the value is the numerical
            encoding of its label
        """
        
        # For each new label
        for idx in label_nums.keys():
        
            # Get new label
            label_num = label_nums[idx]
            
            # Get label from its numerical encoding
            label = self.classes[label_num]
            
            # Update dataset with label and its encoding
            self.df_data.loc[idx, self.label_col] = label
            self.df_data.loc[idx, self.targets_col] = label_num
    
    def get_snapshot(self):
        """
        Returns the dataframe that defines
        the dataset
        """
        
        return self.df_data.copy()
        
        
def pytorch_identify_species(df_img, root_dir, engine, mean, std, batch_size = 32):
    """
    Identifies the species of the animals in all images of a directory
    and adds the species labels to a dataframe that contains the relative
    file paths of all images
    
    df_img (Dataframe) : must contain a column called 'cropped_image_path'
        that has the relative file path of all images in image dir.
        Must not have a column called 'full_path'
    
    root_dir (Path or String): Path to root directory that df_crop's file paths
        are relative to
    
    batch_size (int): Batch size for both embedding and classifier models
    
    """
    
    # Validate parameters
    assert not ('full_path' in df_img.columns)
    df_img = df_img.copy()
    
    # Set up image data transformer
    transform_list = []
    transform_list.append(Resize((256, 256)))
    transform_list.append(CenterCrop((224, 224)))
    transform_list.append(ToTensor())
    transform_list.append(Normalize(mean, std))
    transform = Compose(transform_list)
    
    # Build dataset
    dataset = CSVImageDataset(
        data = df_img.reset_index(drop = True),
        img_dir = root_dir,
        path_col = config.CROP_COL_NAME,
        classes = config.SPECIES_CATEGORIES.tolist(),
        transform = transform
    )
    
    # Get labels
    labels = engine.predict(
        dataloader = dataset.get_data_loader(batch_size = batch_size),
        normalise = True,
        show_progress = False
    )
    
    # Add labels to dataset
    dataset.update_labels(label_nums = dict(zip(range(len(dataset)), labels)))
    
    # Get dataset as dataframe
    df_labels = dataset.get_snapshot()
    
    # Reformat dataset dataframe
    df_labels.drop(dataset.targets_col, inplace = True, axis = 1)
    df_labels.rename(columns = {dataset.label_col : config.SPECIES_COL_NAME}, inplace = True)
    
    # Return df_img with labels
    return df_labels