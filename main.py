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
num_per_study = 20
for key in dcm_study_dict:
    input_list = dcm_study_dict[key]
    med_index = len(input_list) // 2
    # get the median 50 elements
    i1 = int(med_index - num_per_study // 2)
    i2 = int(med_index + num_per_study // 2)
    median_indices += input_list[i1:i2]

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
    unique_ids, test_size=0.2, random_state=464)
train_ids, valid_ids = model_selection.train_test_split(
    train_ids, test_size=0.25, random_state=464)

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

    def __init__(self, t_abs_dcm_files, t_labels, t_shape, batch_size):
        self.abs_dcm_files = t_abs_dcm_files
        self.labels = t_labels
        self.shape = t_shape
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
        x_array = dcm_utils.dcm_images_to_np3d(x_images, self.shape)
        batch_x = dcm_utils.unit_dcm_image(x_array)

        y_labels = self.labels[i1:i2]
        batch_y = np.array(y_labels)

        return batch_x, batch_y


# one-hot encoding using keras' numpy-related utilities
Y_train = np_utils.to_categorical(y_train, n_classes)
Y_valid = np_utils.to_categorical(y_valid, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)
print("Shape after one-hot encoding: ", Y_train.shape)


i_shape = (128, 128)
i_shape3 = (128, 128, 1)
X_train_gen = storage_data_generator(X_train_files, Y_train, i_shape, 32)
X_valid_gen = storage_data_generator(X_valid_files, Y_valid, i_shape, 32)
X_test_gen = storage_data_generator(X_test_files, Y_test, i_shape, 32)


# building a linear stack of layers with the sequential model
model = Sequential()

# convolutional layer
model.add(Conv2D(50, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu',
                 input_shape=i_shape3))

# convolutional layer
model.add(Conv2D(75, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu',
                 kernel_regularizer=keras.regularizers.l2(l=0.01)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(125, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu',
                 kernel_regularizer=keras.regularizers.l2(l=0.01)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# flatten output of conv
model.add(Flatten())

# hidden layer
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(250, activation='relu'))
model.add(Dropout(0.3))
# output layer
model.add(Dense(n_classes, activation='softmax'))

# compiling the sequential model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# training the model for 10 epochs
model.fit_generator(
    generator=X_train_gen,
    validation_data=X_valid_gen,
    steps_per_epoch=len(X_train_gen),
    epochs=10)

prediction = model.predict(X_valid)
scale_pred = []

for pred in prediction:
    if pred[0] > pred[1]:
        scale_pred.append(1)
    else:
        scale_pred.append(2)

# group, predicted, actual
black_correct = 0
black_total = 0
white_correct = 0
white_total = 0

for i in range(500):
    if z_valid[i] == 'black':
        black_total += 1
        if scale_pred[i] == y_valid[i]:
            black_correct += 1
    else:
        white_total += 1
        if scale_pred[i] == y_valid[i]:
            white_correct += 1

black_correct / black_total
white_correct / white_total