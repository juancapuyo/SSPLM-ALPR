import os
import pandas as pd
import torch
from torchvision.io import read_image

class RodoSolDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, annotations_file='annotations.csv', transform=None, target_transform=None):
        self.data_dir = data_dir
        self.annotations_file = annotations_file
        self.transform = transform
        self.target_transform = target_transform

        if os.path.exists(self.annotations_file):
            # Load preprocessed annotations, prevents from having to parse all labels more than once.
            self.img_labels = pd.read_csv(self.annotations_file)
            # Convert 'corners' string back to list of tuples
            self.img_labels['corners'] = self.img_labels['corners'].apply(eval)
        else:
            # Create annotations and save to file
            self.img_labels = self.create_annotations()
            self.img_labels.to_csv(self.annotations_file, index=False)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_labels.iloc[idx]['image']
        image = read_image(img_path)
        label = self.img_labels.iloc[idx].to_dict()
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
    def create_annotations(self):
        """Create a Pandas dataframe with labels and image file paths from label files in data_dir.

        Returns:
            _type_: pd.DataFrame
        """
        annotations = []
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.txt'):
                    img_file = file.replace('.txt', '.jpg')
                    img_path = os.path.join(root, img_file)
                    label_path = os.path.join(root, file)
                    if os.path.exists(img_path):
                        label_info = self.parse_label_file(label_path)
                        label_info['image'] = img_path
                        annotations.append(label_info)
        return pd.DataFrame(annotations)

    def parse_label_file(self, label_path):
        """Parse a label file and return a dictionary with the label information.

        Args:
            string: label_path

        Returns:
            pd.Dataframe : label_info
        """
        label_info = {}
        with open(label_path, 'r') as label_file:
            for line in label_file:
                line = line.strip()
                if not line:
                    continue
                key, value = line.split(': ', 1)
                if key == 'corners':
                    corners = [tuple(map(int, coord.split(','))) for coord in value.split()]        # Corner labels format: 'x1,y1 x2,y2 x3,y3 x4,y4'
                    label_info[key] = corners
                else:
                    label_info[key] = value
        return label_info