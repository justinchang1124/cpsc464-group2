import os
import numpy as np

abs_proj_path = 'C:/Users/justin/PycharmProjects/cpsc464-group2'
groups_path = 'groups.txt'
labels_path = 'labels.txt'
ids_path = 'patient_ids.txt'
abs_ids_path = '{}/{}'.format(abs_proj_path, ids_path)
abs_groups_path = '{}/{}'.format(abs_proj_path, groups_path)
abs_labels_path = '{}/{}'.format(abs_proj_path, labels_path)

groups = np.genfromtxt(abs_groups_path, dtype=int)
ids = np.genfromtxt(abs_ids_path, dtype=str)
labels = np.genfromtxt(abs_labels_path, dtype=int)

group_mapping = ['N/A', 'white', 'black', 'asian', 'native', 'hispanic', 'multi', 'hawa', 'amer indian']