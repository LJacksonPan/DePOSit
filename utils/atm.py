import os
import pandas as pd
import torch
from torch.utils.data import Dataset

class VRDataset(Dataset):
    """
    A dataset class for loading VR data from CSV files.
    """
    def __init__(self, data_dir, transform=None):
        """
        Initializes the dataset object.
        :param data_dir: Directory where the CSV files are located.
        :param transform: Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
        self.data_frames = [pd.read_csv(f) for f in self.file_paths]

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.data_frames)

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset.
        :param idx: Index of the sample to retrieve.
        """
        data_frame = self.data_frames[idx]
        # Extract Main Camera and Controllers' positions and quaternions
        features = data_frame[['MainCam_x', 'MainCam_y', 'MainCam_z',
                               'Quaternion_x', 'Quaternion_y', 'Quaternion_z', 'Quaternion_w',
                               'LeftController_x', 'LeftController_y', 'LeftController_z',
                               'RightController_x', 'RightController_y', 'RightController_z']].values
        # Note: Quaternion columns are repeated in the CSV for left and right controllers, 
        # adjust as necessary to correctly extract these values.
        labels = data_frame[['DeltaTime']].values
        # Convert to tensor
        features = torch.tensor(features, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.float)
        # Apply transformations if any
        if self.transform:
            features = self.transform(features)
            labels = self.transform(labels)
        return features, labels

# Example usage
if __name__ == "__main__":
    dataset = VRDataset(data_dir='path/to/your/csv/files')
    for features, labels in dataset:
        print(features, labels)
