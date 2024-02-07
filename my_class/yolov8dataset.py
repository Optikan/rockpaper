import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np
import random
import yaml

def index_to_class_name(index, class_names):
    """
    Map an index to a class name.

    Args:
        index (int): The index to be mapped.
        class_names (list): A list of class names.

    Returns:
        str: The corresponding class name.
    """
    if 0 <= index < len(class_names):
        return class_names[index]
    else:
        return None

def class_name_to_index(class_name, class_names):
    """
    Map a class name to an index.

    Args:
        class_name (str): The class name to be mapped.
        class_names (list): A list of class names.

    Returns:
        int: The corresponding index.
    """
    try:
        return class_names.index(class_name)
    except ValueError:
        return None

class Sample:
    def __init__(self, image, class_id, class_name, boxes):
        self.image = image
        self.class_id = class_id
        self.class_name = class_name
        self.boxes = boxes

        # Create a color map dynamically based on the number of unique class IDs
        unique_class_ids = np.unique(class_id)
        self.color_map = {class_id: colormaps.get_cmap('hsv')(i/len(unique_class_ids)) for i, class_id in enumerate(unique_class_ids)}
    
    def __repr__(self):
        return f'Sample(image={self.image}, class_id={self.class_id}, class_name={self.class_name}, boxes={self.boxes})'
    
    def display_image(self, show_boxes=True):
        image = np.array(self.image)
        
        # Plot the original image
        plt.imshow(image)
        
        if show_boxes or len(self.boxes) > 0:
        # Calculate the actual x, y coordinates of each point based on the image dimensions
            height, width, _ = image.shape
            for class_id, class_name, box in zip(self.class_id, self.class_name, self.boxes):
                x_normalized, y_normalized, w_normalized, h_normalized = box
                x = int(x_normalized * width)
                y = int(y_normalized * height)
                w = int(w_normalized * width)
                h = int(h_normalized * height)
                
                # Retrieve the color based on the class ID
                color = self.color_map.get(class_id, 'black')  # Default to black if class ID not found
                
                # Draw a rectangle representing the detected object using its width and height
                # The rectangle will be drawn with its top-left corner at (x - w/2, y - h/2)
                rect = plt.Rectangle((x - w/2, y - h/2), w, h, fill=False, color=color, linewidth=2)
                plt.gca().add_patch(rect)
                
                # Add the class name as text near the top-left corner of the rectangle
                plt.text(x - w/2, y - h/2, class_name, fontsize=10, color=color, verticalalignment='top')
        
        plt.axis('off')
        plt.show()


class Yolov8SubDataset(Dataset):
    def __init__(self, sub_dir, transform=None, class_attributes=None):
        self.sub_dir = sub_dir
        self.transform = transform
        self.class_attributes = class_attributes
        

        if self.class_attributes is None:
            raise ValueError("class_attributes must be provided.")


        self.image_dir = os.path.join(sub_dir, 'images')
        self.labels_dir = os.path.join(sub_dir, 'labels')

        # Get the list of image file names and corresponding labels
        self.images = os.listdir(self.image_dir)
        self.labels = os.listdir(self.labels_dir)


        self.image_counts = self.count_images(self.image_dir)
        self.class_counts = self.calculate_class_counts()

        self.nb_images_without_labels,self.without_label_indices = self.count_images_without_labels()
        

    def calculate_class_counts(self):
        class_counts = {class_id: 0 for class_id, name in enumerate(self.class_attributes)}
        for sample in self:
            for class_id in sample.class_id:
                class_counts[class_id] += 1
        return class_counts

    def count_images(self, dir_path):
        if os.path.exists(dir_path):
            return len(os.listdir(dir_path))
        else:
            return 0

    def count_images_without_labels(self):
        nb_images_without_labels = 0
        without_label_indices = []
        for index,label_name in enumerate(self.labels):
            label_name = os.path.join(self.labels_dir, self.labels[index])
            with open(label_name, 'r') as file:
                lines = file.readlines()
                if len(lines) == 0:
                    # print(f"Label file '{label_name}' is empty.")
                    nb_images_without_labels += 1
                    without_label_indices.append(index)
        return nb_images_without_labels, without_label_indices
        
        

    
    def get_random_image_without_label(self):
        images_without_labels = [self[index] for index in self.without_label_indices]
        
        if images_without_labels:
            random_image = random.choice(images_without_labels)

            return random_image
        else:
            print("No images without labels found.")
            return None



    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.images[idx])
        label_name = os.path.join(self.labels_dir, self.labels[idx])

        # Load image
        image = Image.open(img_name)

        # Parse label file
        class_ids = []
        class_names = []
        boxes = []
        with open(label_name, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if line.strip() != '':
                    label = line.strip().split()
                    class_id = int(label[0])
                    class_ids.append(class_id)
                    class_name = self.class_attributes[class_id]
                    class_names.append(class_name)
                    bbox = list(map(float, label[1:]))
                    boxes.append(bbox)

        return Sample(image, class_ids, class_names, boxes)

    def __repr__(self):
        len_info = f"Number of samples: {len(self)}"
        class_info = "\n".join([f"- {index_to_class_name(class_id, self.class_attributes)}: {count} samples" for
                                class_id, count in self.class_counts.items()])
        class_info += f"\n- without labels: {self.nb_images_without_labels} samples"
        return f"Subdirectory: {self.sub_dir}\n{len_info} \nClass Counts:\n{class_info}\nClass Attributes: {self.class_attributes}"

    def get_random_sample_by_class_id(self, class_id):
        matching_indices = [i for i, sample in enumerate(self) if class_id in sample.class_id]
        if not matching_indices:
            print(f"No samples found with class ID '{class_id}'")
            return None
        random_index = random.choice(matching_indices)
        return self[random_index]



class Yolov8Dataset:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.sub_datasets = {}
       
        self.data_yaml_path = os.path.join(root_dir, 'data.yaml')


         # Get the list of first-level subdirectories within the root directory
        self.sub_dirs = [sub_dir for sub_dir in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, sub_dir))]
        

        # Load class attribute names from data.yaml
        with open(self.data_yaml_path, 'r') as file:
            self.class_attributes = list(yaml.safe_load(file)['names'].values())
            print(f"Class attributes: {self.class_attributes}")

        self.load_sub_datasets()

    def __len__(self):
        return sum([len(sub_dataset) for sub_dataset in self.sub_datasets.values()])
 

    def load_sub_datasets(self):
    
        for sub_dir in self.sub_dirs:
            self.sub_datasets[sub_dir] = Yolov8SubDataset(os.path.join(self.root_dir, sub_dir), class_attributes=self.class_attributes)

    def __getitem__(self, sub_dir):
        assert sub_dir in self.sub_datasets, f"Subdirectory '{sub_dir}' not found. Please choose from the following: {self.sub_dirs}"
        return self.sub_datasets[sub_dir]
    
    def __repr__(self):
        repr_str = f"Dataset : {self.root_dir} \n Number of samples: {len(self)}\n\n"
        for sub_dir, sub_dataset in self.sub_datasets.items():
            repr_str += f"{sub_dataset.__repr__()}\n\n"

        return repr_str
   

