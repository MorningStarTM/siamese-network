import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import random
import numpy as np



class SiameseNetworkDataset(Dataset):
    def __init__(self, stage_1, stage_2, stage_3, transform=None):
        """
        Initialize the dataset with lists of image paths for each class.
        
        Args:
            stage_1 (list of str): List of image paths for class 1.
            stage_2 (list of str): List of image paths for class 2.
            stage_3 (list of str): List of image paths for class 3.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.class_images = {
            'stage_1': stage_1,
            'stage_2': stage_2,
            'stage_3': stage_3
        }
        self.transform = transform
        self.x0, self.x1, self.labels = self.create_pairs()
        self.size = len(self.labels)

    def create_pairs(self):
        x0_data = []
        x1_data = []
        labels = []

        n = min([len(self.class_images[c]) for c in self.class_images]) - 1
        classes = list(self.class_images.keys())
        
        for cls in classes:
            images = self.class_images[cls]
            for i in range(n):
                # Create positive pair
                img1, img2 = images[i], images[i + 1]
                x0_data.append(img1)
                x1_data.append(img2)
                labels.append(1)

                # Create negative pair
                other_cls = random.choice([c for c in classes if c != cls])
                img3 = self.class_images[other_cls][i]
                x0_data.append(img1)
                x1_data.append(img3)
                labels.append(0)

        labels = np.array(labels, dtype=np.int32)
        return x0_data, x1_data, labels

    def __getitem__(self, index):
        img0_path = self.x0[index]
        img1_path = self.x1[index]
        label = torch.tensor(self.labels[index], dtype=torch.float32)

        img0 = Image.open(img0_path).convert('RGB')
        img1 = Image.open(img1_path).convert('RGB')

        if self.transform:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return (img0, img1), label

    def __len__(self):
        return self.size

    def plot_image_pair(self, index):
        """
        Plot a pair of images with their label.

        Args:
            index (int): The index of the image pair to plot.
        """
        img0_path = self.x0[index]
        img1_path = self.x1[index]
        label = self.labels[index]

        img0 = Image.open(img0_path).convert('RGB')
        img1 = Image.open(img1_path).convert('RGB')

        if self.transform:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        if torch.is_tensor(img0):
            img0 = img0.permute(1, 2, 0).numpy()
        if torch.is_tensor(img1):
            img1 = img1.permute(1, 2, 0).numpy()

        _, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(img0)
        ax[0].axis('off')
        ax[0].set_title("Image 1")

        ax[1].imshow(img1)
        ax[1].axis('off')
        ax[1].set_title(f"Image 2\nLabel: {'Positive' if label == 1 else 'Negative'}")

        plt.show()

    @staticmethod
    def create_dataloader(stage_1, stage_2, stage_3, batch_size, transform=None, shuffle=False):
        """
        Create a DataLoader for the SiameseNetworkDataset.

        Args:
            stage_1 (list of str): List of image paths for class 1.
            stage_2 (list of str): List of image paths for class 2.
            stage_3 (list of str): List of image paths for class 3.
            batch_size (int): Number of samples per batch.
            transform (callable, optional): Optional transform to be applied on a sample.
            shuffle (bool): Whether to shuffle the dataset.

        Returns:
            DataLoader: DataLoader for the SiameseNetworkDataset.
        """
        dataset = SiameseNetworkDataset(stage_1, stage_2, stage_3, transform=transform)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)




class SiameseTripletDataset(Dataset):
    def __init__(self, stage_1, stage_2, stage_3, transform=None):
        """
        Initialize the dataset with lists of image paths for each class.
        
        Args:
            stage_1 (list of str): List of image paths for class 1.
            stage_2 (list of str): List of image paths for class 2.
            stage_3 (list of str): List of image paths for class 3.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.class_images = {
            'stage_1': stage_1,
            'stage_2': stage_2,
            'stage_3': stage_3
        }
        self.transform = transform
        self.triplets = self.create_triplets()
        self.size = len(self.triplets)

    def create_triplets(self):
        triplets = []
        n = min([len(self.class_images[c]) for c in self.class_images]) - 1
        classes = list(self.class_images.keys())
        
        for cls in classes:
            images = self.class_images[cls]
            for i in range(n):
                anchor = images[i]
                positive = images[i + 1]
                other_cls = random.choice([c for c in classes if c != cls])
                negative = random.choice(self.class_images[other_cls])
                triplets.append((anchor, positive, negative))
                
        return triplets

    def __getitem__(self, index):
        anchor_path, positive_path, negative_path = self.triplets[index]
        
        anchor = Image.open(anchor_path).convert('RGB')
        positive = Image.open(positive_path).convert('RGB')
        negative = Image.open(negative_path).convert('RGB')

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return (anchor, positive, negative), 0  # 0 is a dummy label as it's not used with triplet loss

    def __len__(self):
        return self.size

    def plot_triplet(self, index):
        """
        Plot a triplet of images (anchor, positive, negative).

        Args:
            index (int): The index of the triplet to plot.
        """
        anchor_path, positive_path, negative_path = self.triplets[index]
        print(f"Anchor: {anchor_path}")
        print(f"Positive: {positive_path}")
        print(f"Negative: {negative_path}")
        anchor = Image.open(anchor_path).convert('RGB')
        positive = Image.open(positive_path).convert('RGB')
        negative = Image.open(negative_path).convert('RGB')

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        if torch.is_tensor(anchor):
            anchor = anchor.permute(1, 2, 0).numpy()
        if torch.is_tensor(positive):
            positive = positive.permute(1, 2, 0).numpy()
        if torch.is_tensor(negative):
            negative = negative.permute(1, 2, 0).numpy()

        _, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(anchor)
        ax[0].axis('off')
        ax[0].set_title("Anchor")

        ax[1].imshow(positive)
        ax[1].axis('off')
        ax[1].set_title("Positive")

        ax[2].imshow(negative)
        ax[2].axis('off')
        ax[2].set_title("Negative")

        plt.show()

    @staticmethod
    def create_dataloader(stage_1, stage_2, stage_3, batch_size, transform=None, shuffle=False):
        """
        Create a DataLoader for the SiameseNetworkDataset.

        Args:
            stage_1 (list of str): List of image paths for class 1.
            stage_2 (list of str): List of image paths for class 2.
            stage_3 (list of str): List of image paths for class 3.
            batch_size (int): Number of samples per batch.
            transform (callable, optional): Optional transform to be applied on a sample.
            shuffle (bool): Whether to shuffle the dataset.

        Returns:
            DataLoader: DataLoader for the SiameseNetworkDataset.
        """
        dataset = SiameseNetworkDataset(stage_1, stage_2, stage_3, transform=transform)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)