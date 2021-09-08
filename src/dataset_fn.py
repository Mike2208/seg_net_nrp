import torch.utils.data as data
import numpy as np
# import torchvideo.transforms as VT
import torchvision.transforms as IT
import torchvision.transforms.functional as TF
import PIL
import warnings
import os
import random
import torch
from PIL import Image
VGG_MEAN = [0.0, 0.0, 0.0]
VGG_STD = [1.0, 1.0, 1.0]


class Mots_Dataset(data.Dataset):
  def __init__(self, root_dir, train_valid_subfolders, n_classes, n_frames, remove_ground):
    self.sample_root = os.path.join(root_dir, 'training', 'image_02')
    self.label_root = os.path.join(root_dir, 'instances')
    self.train_valid_subfolders = train_valid_subfolders
    self.n_classes = n_classes
    self.n_frames = n_frames  # idea: variable n_frames for each batch?
    self.remove_ground = remove_ground
    # self.sample_transform = IT.Compose([
    #   IT.RandomResizedCrop(size=(224, 224)),
    #   IT.ToTensor()])
    # IT.Normalize(mean=VGG_MEAN, std=VGG_STD)
    self.label_transform = IT.ToTensor()

  def transform(self, list_of_images):
    """
    Parameters
    ----------
    list_of_images : list
      Must be a list of tuples, where each tuple inside the list contains both a training image
      and its corresponding label.

    Returns
    -------
    transformed_images : list
      A list of tuples, where each tuple is a set of (image, label) for a specific timepoint.
    """
    transformed_images = []
    resize = IT.transforms.Resize(size=(224, 224), interpolation=IT.InterpolationMode.NEAREST)
    i, j, h, w = IT.transforms.RandomCrop.get_params(list_of_images[0][0], output_size=(350, 350))
    for image, label in list_of_images:
      image, label = TF.crop(image, i, j, h, w), TF.crop(label, i, j, h, w)
      image, label = resize(image), resize(label)
      image, label = TF.to_tensor(np.array(image)), TF.to_tensor(np.array(label))
      transformed_images.append((image, label))
    return transformed_images

  def __len__(self):
    return len(self.train_valid_subfolders)

  def __getitem__(self, index):
    sample_dir = os.path.join(self.sample_root, f'{index:04}')
    label_dir = os.path.join(self.label_root, f'{index:04}')
    sample_paths = [os.path.join(sample_dir, p) for p in os.listdir(sample_dir)]
    label_paths = [os.path.join(label_dir, p) for p in os.listdir(label_dir)]
    n_frames = min(self.n_frames, len(sample_paths) - 1)
    first_frame = random.choice(np.arange(len(sample_paths) - n_frames))
    sample_frames = sample_paths[first_frame:first_frame + n_frames]
    label_frames = label_paths[first_frame:first_frame + n_frames]
    # for frame in sample_paths:
    # samples_and_labels = self.transform([(Image.open(os.path.join(sample_dir, p)), Image.open(os.path.join(label_dir, p))) for p in sample_frames])
    samples_and_labels = []
    for i in range(n_frames):
      sample = Image.open(os.path.join(sample_dir, sample_frames[i]))
      label = Image.open(os.path.join(label_dir, label_frames[i]))
      g = np.unique(np.array(label))
      samples_and_labels.append((sample, label))
    samples_and_labels = self.transform(samples_and_labels)

    samples = torch.stack([item[0] for item in samples_and_labels], dim=-1)
    labels = torch.stack([item[1] for item in samples_and_labels], dim=-1)
    # if self.remove_ground:
    #   labels[labels == 10000] = 0
    #   warnings.warn('UserWarning: by setting remove_background = True, if you are using the MOTS dataset, you are'
    #                 ' collapsing the background class with the ignore region class.')
    y = torch.unique(labels)
    labels = torch.div(labels, 1000, rounding_mode='floor')
    z = torch.unique(labels)
    labels[labels == 10] = self.n_classes - int(not self.remove_ground)
    x = torch.unique(labels)
    labels = torch.nn.functional.one_hot(labels.to(torch.int64), num_classes=self.n_classes + int(self.remove_ground))
    labels = torch.movedim(labels, -1, 1)
    labels = torch.squeeze(labels, dim=0)
    labels = labels.type(torch.FloatTensor)
    if self.remove_ground:
      labels = labels[1:]
    #sample = self.sample_transform(Image.open(os.path.join(sample_dir, frame)))
    # labels = [self.transform(Image.open(os.path.join(label_dir, p))) for p in sample_frames] # espescially output dims? might have to use permute
    # labels[labels == 10] = 0

    return samples.to(device='cuda'), labels.to(device='cuda')


def get_datasets_seg(root_dir, train_valid_ratio,
                     batch_size_train, batch_size_valid, n_frames,
                     augmentation, n_classes, speedup_factor, remove_ground):
  
  # Data train valid
  training_folders = os.listdir(os.path.join(root_dir, 'training', 'image_02'))
  train_subfolders = [folder for folder in training_folders[:int(train_valid_ratio * len(training_folders))]]
  valid_subfolders = [folder for folder in training_folders[int(train_valid_ratio * len(training_folders)):]]

  # Training dataloader
  train_dataset = Mots_Dataset(root_dir, train_subfolders, n_classes, n_frames, remove_ground)
  train_dataloader = data.DataLoader(
    train_dataset, batch_size=batch_size_train, shuffle=True)

  # Validation dataloader
  valid_dataset = Mots_Dataset(root_dir, valid_subfolders, n_classes, n_frames, remove_ground)
  valid_dataloader = data.DataLoader(
    valid_dataset, batch_size=batch_size_valid, shuffle=True)

  # Return the dataloaders to the computer
  return train_dataloader, valid_dataloader
