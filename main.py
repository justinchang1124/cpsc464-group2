import random
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.utils import np_utils, Sequence
import os
import numpy as np
from sklearn import model_selection
import keras.regularizers
import keras

# project code
import dcm_utils
import read_metadata
from labels_presentation import *

abs_proj_path = 'C:/Users/justin/PycharmProjects/cpsc464-group2'
data_path = 'image_data/manifest-1654812109500/Duke-Breast-Cancer-MRI'
abs_data_path = os.path.join(abs_proj_path, data_path)
dcm_files = dcm_utils.dcm_dir_list(abs_data_path)
n_dcm_files = len(dcm_files)

dcm_files_data = []
for file in dcm_files:
    dcm_files_data.append(file.split('\\'))

dcm_study_dict = {}
for i in range(n_dcm_files):
    key = os.path.join(*dcm_files_data[i][:-1])
    input_list = dcm_study_dict.get(key)
    if input_list is None:
        input_list = []
    input_list.append(i)
    dcm_study_dict[key] = input_list

median_indices = []
for key in dcm_study_dict:
    input_list = dcm_study_dict[key]
    # med_index = len(input_list) // 2
    # num_per_study = 40
    # get the median 50 elements
    i1 = 0  # int(med_index - num_per_study // 2)
    i2 = len(input_list)  # int(med_index + num_per_study // 2)
    sep_len = 8
    median_indices += input_list[i1:i2:sep_len]  # space out b/c we don't want too-similar images

# truncate
tn_dcm_files = len(median_indices)
# t_dcm_files = []
t_dcm_files_data = []
for i in median_indices:
    # t_dcm_files.append(dcm_files[i])
    t_dcm_files_data.append(dcm_files_data[i])


# open metadata
global_ids = []
for i in range(tn_dcm_files):
    global_ids.append(t_dcm_files_data[i][0])
labels, groups = read_metadata.get_labels_groups(global_ids)
abs_dcm_files = dcm_utils.prepend_abs(abs_data_path, dcm_files)
n_classes = len(set(labels))

for i in range(tn_dcm_files):
    if labels[i] < 0:
        labels[i] = 0

# partition into train / test / split
unique_ids = list(set(global_ids))
train_ids, test_ids = model_selection.train_test_split(
    unique_ids, test_size=0.2) # random_state?
train_ids, valid_ids = model_selection.train_test_split(
    train_ids, test_size=0.25)

X_train_files = []
X_valid_files = []
X_test_files = []
y_train = []
y_valid = []
y_test = []
z_train = []
z_valid = []
z_test = []

for i in range(tn_dcm_files):
    gid = global_ids[i]
    if gid in train_ids:
        X_train_files.append(abs_dcm_files[i])
        y_train.append(labels[i])
        z_train.append(groups[i])
    if gid in valid_ids:
        X_valid_files.append(abs_dcm_files[i])
        y_valid.append(labels[i])
        z_valid.append(groups[i])
    if gid in test_ids:
        X_test_files.append(abs_dcm_files[i])
        y_test.append(labels[i])
        z_test.append(groups[i])


# note: fears of symmetry, maybe need to make the generator more robust?


# loading the dataset
class storage_data_generator(Sequence):

    def __init__(self, t_abs_dcm_files, t_labels, batch_size):
        # randomize to avoid generating adjacent ones in the correct order
        files_labels = list(zip(t_abs_dcm_files, t_labels))
        random.shuffle(files_labels)
        self.abs_dcm_files, self.labels = zip(*files_labels)
        self.batch_size = batch_size

    def __len__(self):
        n_samp = len(self.abs_dcm_files)
        n_batch = np.ceil(n_samp / float(self.batch_size))
        return n_batch.astype(np.int)

    def __getitem__(self, idx):
        i1 = idx * self.batch_size
        i2 = (idx + 1) * self.batch_size

        x_files = self.abs_dcm_files[i1:i2]
        x_images = dcm_utils.open_dcm_images(x_files)
        x_array = dcm_utils.dcm_images_to_np3d(x_images)
        batch_x = x_array  # dcm_utils.unit_dcm_image(x_array)

        y_labels = self.labels[i1:i2]
        batch_y = np.array(y_labels)

        return batch_x, batch_y


# one-hot encoding using keras' numpy-related utilities
Y_train = np_utils.to_categorical(y_train, n_classes)
Y_valid = np_utils.to_categorical(y_valid, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)
print("Shape after one-hot encoding: ", Y_train.shape)


glob_batch_size = 8
X_train_gen = storage_data_generator(X_train_files, Y_train, glob_batch_size)
X_valid_gen = storage_data_generator(X_valid_files, Y_valid, glob_batch_size)
X_test_gen = storage_data_generator(X_test_files, Y_test, glob_batch_size)


i_shape = (128, 128)
i_shape3 = (*i_shape, 1)

# building a linear stack of layers with the sequential model
model = Sequential()

# convolutional layer (originally 50, 75, 125, 0.25 dropout for both)
model.add(Conv2D(32, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu',
                 input_shape=i_shape3,
                 kernel_regularizer=keras.regularizers.l2(l=0.01)))

# convolutional layer
model.add(Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu',
                 kernel_regularizer=keras.regularizers.l2(l=0.01)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.1))

model.add(Conv2D(96, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu',
                 kernel_regularizer=keras.regularizers.l2(l=0.01)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.1))

# flatten output of conv
model.add(Flatten())

# hidden layer
model.add(Dense(125, activation='relu')) # originally 500, 250; 0.4, 0.3
model.add(Dropout(0.5))
model.add(Dense(125, activation='relu'))
model.add(Dropout(0.5))
# output layer
model.add(Dense(n_classes, activation='softmax'))

# compiling the sequential model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
print(model.summary())

# training the model for 10 epochs
model.fit_generator(
    generator=X_train_gen,
    validation_data=X_valid_gen,
    steps_per_epoch=len(X_train_gen),
    epochs=10)

# note: regularization seems to have a large benefit!
prediction = model.predict_generator(generator=X_test_gen)
scale_pred = []

for pred in prediction:
    scale_pred.append(np.argmax(pred))

ea_dict, er_dict = separate_by_group(scale_pred, y_test, z_test)
print(ea_dict, er_dict)
summarize_ar_dict(ea_dict, er_dict)




