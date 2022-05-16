import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from PIL import Image

dataset_path = r'D:\DL\datasets\nrp'
dataset_type = 'multi_shelf'
train_path = os.path.join(dataset_path, dataset_type, 'train.hdf5')
valid_path = os.path.join(dataset_path, dataset_type, 'valid.hdf5')
img_dir = os.path.join(dataset_path, dataset_type, 'camera_images')
seg_dir = os.path.join(dataset_path, dataset_type, 'segment_images')
label_id_map = {'a_marbles': 0, 'b_marbles': 1, 'apple': 2, 'banana': 3}

n_sequences_train = 1600  # 1172  # 188  # 188
n_sequences_valid = 400  # 293  # 47  # 47
n_frames_max = 59  # 55
h, w = 320, 320
n_channels = 3
n_classes = max(label_id_map.values()) + 1
no_hot_labels = False

show_dataset_mode = False
if show_dataset_mode:
    with h5py.File(train_path, 'r') as f:
        samples = np.array(f['samples'])
        labels = np.array(f['labels'])
    for frame_id in range(n_frames_max):
        plt.imshow(labels[0, 2, :, :, frame_id])
        plt.show()
    exit()
else:
    f_train = h5py.File(train_path, 'w')
    f_valid = h5py.File(valid_path, 'w')

img_train_array_shape = (n_sequences_train, n_channels, h, w, n_frames_max)
img_train_array = np.zeros(img_train_array_shape, dtype=np.uint8)
img_valid_array_shape = (n_sequences_valid, n_channels, h, w, n_frames_max)
img_valid_array = np.zeros(img_valid_array_shape, dtype=np.uint8)
prev_sequence_id = -1
for img_path in os.listdir(img_dir):
    sequence_id = int(img_path.split('cam_img_')[1].split('_')[0])
    if sequence_id < n_sequences_train + n_sequences_valid:
        if prev_sequence_id != sequence_id:
            frame_id = 0
            prev_sequence_id = sequence_id
            print(f'\rDoing image sequence {sequence_id:04}', end='')
        if frame_id < n_frames_max:
            img_full_path = os.path.join(img_dir, img_path)
            with Image.open(img_full_path) as read_img:
                to_write = np.array(read_img).transpose((2, 0, 1))
                if sequence_id < n_sequences_train:
                    img_train_array[sequence_id, :, :, :, frame_id] = to_write
                else:
                    valid_id = sequence_id - n_sequences_train
                    img_valid_array[valid_id, :, :, :, frame_id] = to_write
            frame_id += 1

f_train.create_dataset('samples', img_train_array_shape, data=img_train_array)
f_valid.create_dataset('samples', img_valid_array_shape, data=img_valid_array)
del img_train_array, img_valid_array

seg_train_array_shape = (n_sequences_train, 1, h, w, n_frames_max)
seg_train_array = np.zeros(seg_train_array_shape, dtype=np.uint8)
seg_valid_array_shape = (n_sequences_valid, 1, h, w, n_frames_max)
seg_valid_array = np.zeros(seg_valid_array_shape, dtype=np.uint8)
prev_sequence_id = -1
prev_frame_number = -1
for seg_path in os.listdir(seg_dir):
    sequence_id = int(seg_path.split('seg_img_')[1].split('_')[0])
    frame_number = int(seg_path.split('seg_img_')[1].split('_')[1])
    if sequence_id < n_sequences_train + n_sequences_valid:
        # label_name = seg_path.split('__')[-1].split('.png')[0]
        label_name = '_'.join(seg_path.split('__')[0].split('_')[4:])
        label_id = label_id_map[label_name]
        if prev_sequence_id != sequence_id:
            prev_sequence_id = sequence_id
            frame_id = -1
            print(f'\rDoing label sequence {sequence_id:04}', end='')
        if prev_frame_number != frame_number:
            prev_frame_number = frame_number
            frame_id += 1
        if frame_id < n_frames_max:
            seg_full_path = os.path.join(seg_dir, seg_path)
            with Image.open(seg_full_path) as read_seg:
                to_write = np.array(read_seg)[:, :, 0]
                to_write = np.where(to_write > 0, 1, 0).astype(np.uint8)
                to_write *= (label_id + 1)
                if sequence_id < n_sequences_train:
                    seg_train_array[sequence_id, :, :, :, frame_id] += to_write
                else:
                    valid_id = sequence_id - n_sequences_train
                    seg_valid_array[valid_id, :, :, :, frame_id] += to_write

if no_hot_labels:
    seg_train_array = np.argmax(seg_train_array, axis=1)
    seg_valid_array = np.argmax(seg_valid_array, axis=1)
    seg_train_array_shape = (n_sequences_train, h, w, n_frames_max)
    seg_valid_array_shape = (n_sequences_valid, h, w, n_frames_max)
f_train.create_dataset('labels', seg_train_array_shape, data=seg_train_array)
f_valid.create_dataset('labels', seg_valid_array_shape, data=seg_valid_array)
f_train.close()
f_valid.close()
