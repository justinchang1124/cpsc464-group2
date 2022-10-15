from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.utils import np_utils, Sequence
import os
import numpy as np
from sklearn import model_selection

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

# open metadata
global_ids = []
for i in range(n_dcm_files):
    global_ids.append(dcm_files_data[i][0])
labels, groups = read_metadata.get_labels_groups(global_ids)
abs_dcm_files = dcm_utils.prepend_abs(abs_data_path, dcm_files)
n_classes = len(set(labels))

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

for i in range(n_dcm_files):
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
        batch_x = dcm_utils.dcm_images_to_np3d(x_images, self.t_shape)

        y_labels = self.labels[i1:i2]
        batch_y = np.array(y_labels)

        return batch_x, batch_y


X_train_gen = storage_data_generator(X_train_files, y_train, (128, 128), 32)
X_valid_gen = storage_data_generator(X_valid_files, y_valid, (128, 128), 32)
X_test_gen = storage_data_generator(X_test_files, y_test, (128, 128), 32)



dcm_images = dcm_utils.open_dcm_images(abs_dcm_files[0:99])
dcm_np3d = dcm_utils.dcm_images_to_np3d(dcm_images, (128, 128))

X_train = np.empty((1000, 128, 128))
X_valid = np.empty((500, 128, 128))
X_test = np.empty((1000, 128, 128))

i_xy = (128, 128)
i_shape = (128, 128, 1)

for i in range(501):
    if i == 180: # weird sample, get rid of it?
        continue
    rind = 5 * i
    lind = i
    if i > 180:
        lind -= 1

    X_train[2*lind] = dcm_utils.resize_dcm_image(dcm_images[rind], i_xy)
    X_train[2*lind + 1] = dcm_utils.resize_dcm_image(dcm_images[rind + 1], i_xy)
    X_valid[lind] = dcm_utils.resize_dcm_image(dcm_images[rind+2], i_xy)
    X_test[2*lind] = dcm_utils.resize_dcm_image(dcm_images[rind + 3], i_xy)
    X_test[2*lind + 1] = dcm_utils.resize_dcm_image(dcm_images[rind + 4], i_xy)

del(dcm_images)

y_train = np.empty(1000)
y_valid = np.empty(500)
y_test = np.empty(1000)
z_train = [None] * 1000
z_valid = [None] * 500
z_test = [None] * 1000

for i in range(501):
    if i == 180: # weird sample, get rid of it?
        continue
    rind = 5 * i
    lind = i
    if i > 180:
        lind -= 1
    y_train[2 * lind] = labels[rind]
    y_train[2 * lind + 1] = labels[rind+1]
    y_valid[lind] = labels[rind+2]
    y_test[2 * lind] = labels[rind+3]
    y_test[2 * lind + 1] = labels[rind+4]
    z_train[2 * lind] = groups[rind]
    z_train[2 * lind + 1] = groups[rind + 1]
    z_valid[lind] = groups[rind + 2]
    z_test[2 * lind] = groups[rind + 3]
    z_test[2 * lind + 1] = groups[rind + 4]


# # building the input vector from the 32x32 pixels
X_train = X_train.reshape(X_train.shape[0], *i_shape)
X_valid = X_valid.reshape(X_valid.shape[0], *i_shape)
X_test = X_test.reshape(X_test.shape[0], *i_shape)
X_train = X_train.astype('float32')
X_valid = X_valid.astype('float32')
X_test = X_test.astype('float32')


# normalizing the data to help with the training
X_train /= 255
X_valid /= 255
X_test /= 255

# one-hot encoding using keras' numpy-related utilities
n_classes = max(labels) - min(labels) + 1
print("Shape before one-hot encoding: ", y_train.shape)
Y_train = np_utils.to_categorical(y_train - 1, n_classes)
Y_valid = np_utils.to_categorical(y_valid - 1, n_classes)
Y_test = np_utils.to_categorical(y_test - 1, n_classes)
print("Shape after one-hot encoding: ", Y_train.shape)

# building a linear stack of layers with the sequential model
model = Sequential()

# convolutional layer
model.add(Conv2D(50, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu',
                 input_shape=i_shape))

# convolutional layer
model.add(Conv2D(75, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(125, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
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
model.add(Dense(2, activation='softmax'))

# compiling the sequential model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# training the model for 10 epochs
model.fit(X_train, Y_train, batch_size=32, epochs=10, validation_data=(X_test, Y_test))

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