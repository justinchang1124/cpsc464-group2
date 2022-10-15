from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.utils import np_utils
import os
import numpy as np

# project code
import image_testing

abs_proj_path = 'C:/Users/justin/PycharmProjects/cpsc464-group2'
data_path = 'image_data/manifest-1654812109500/Duke-Breast-Cancer-MRI'
abs_data_path = os.path.join(abs_proj_path, data_path)
dcm_files = image_testing.dcm_dir_list(abs_data_path)
n_dcm_files = len(dcm_files)

dcm_files_data = []
for file in dcm_files:
    dcm_files_data.append(file.split('\\'))

dcm_studies = [None] * n_dcm_files
for i in range(n_dcm_files):
    dcm_studies[i] = os.path.join(*dcm_files_data[i][:-1]) # splat
dcm_studies = sorted(list(set(dcm_studies)))

# for dcm_study in dcm_studies:
#     print(dcm_study)

for i in range(6):
    dcm_study_examp = dcm_studies[i]
    dcm_images_examp = image_testing.open_dcm_folder(os.path.join(abs_data_path, dcm_study_examp))
    image_testing.animate_dcm_images(dcm_images_examp)


global_ids = []
for i in range(n_dcm_files):
    global_ids.append(dcm_files_data[i][0])

print("IMPORTED: image_testing.py")

# Breast_MRI_001\01-01-1990-NA-MRI BREAST BILATERAL WWO-97538\11.000000-ax dyn 3rd pass-41458
#


# dcm_study_examp = dcm_studies[0]
# dcm_images_examp = open_dcm_images(os.path.join(abs_data_path, dcm_study_examp))
# x1 = animate_dcm_images(dcm_images_examp[0:80])
# plt.show()




# def get_sample_from_dcm_mat(dcm_loc_mat, samp_name):
#     num_matches = 0
#     for i in range(dcm_loc_mat.shape[0]):
#         if dcm_loc_mat[i][0] == samp_name:
#             num_matches += 1
#
#     np_copy = np.empty(shape=(num_matches, dcm_loc_mat.shape[1]), dtype=dcm_loc_mat.dtype)
#     num_matches = 0
#     for i in range(dcm_loc_mat.shape[0]):
#         if dcm_loc_mat[i][0] == samp_name:
#             np_copy[num_matches] = dcm_loc_mat[i]
#             num_matches += 1
#
#     return np_copy



# goal:
# 1. filter down a DCM image set
# 2. filter down

# loading the dataset
labels, groups = get_labels_groups(global_ids)
abs_dcm_files = dcm_dir_list(abs_data_path, abs=True)
dcm_images = open_dcm_images(abs_dcm_files)

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

    X_train[2*lind] = image_testing.resize_dcm_image(dcm_images[rind], i_xy)
    X_train[2*lind + 1] = image_testing.resize_dcm_image(dcm_images[rind+1], i_xy)
    X_valid[lind] = image_testing.resize_dcm_image(dcm_images[rind+2], i_xy)
    X_test[2*lind] = image_testing.resize_dcm_image(dcm_images[rind+3], i_xy)
    X_test[2*lind + 1] = image_testing.resize_dcm_image(dcm_images[rind+4], i_xy)

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