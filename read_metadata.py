import os
import numpy as np

abs_proj_path = 'C:/Users/justin/PycharmProjects/cpsc464-group2'
groups_path = 'groups.txt'
labels_path = 'labels.txt'
ids_path = 'patient_ids.txt'
abs_ids_path = '{}/{}'.format(abs_proj_path, ids_path)
abs_groups_path = '{}/{}'.format(abs_proj_path, groups_path)
abs_labels_path = '{}/{}'.format(abs_proj_path, labels_path)

met_groups = np.genfromtxt(abs_groups_path, dtype=int)
met_ids = np.genfromtxt(abs_ids_path, dtype=str)
met_labels = np.genfromtxt(abs_labels_path, dtype=int)
n = len(met_ids)

group_mapping = ['N/A', 'white', 'black', 'asian', 'native', 'hispanic', 'multi', 'hawa', 'amer indian']
id_to_grp = {}
id_to_lab = {}

for i in range(n):
    id_to_grp[met_ids[i]] = group_mapping[met_groups[i]]
    id_to_lab[met_ids[i]] = met_labels[i]

def get_labels_groups(ids):
    n = len(ids)
    labels = [None] * n
    groups = [None] * n
    for i in range(n):
        id = ids[i]
        labels[i] = id_to_lab[id]
        groups[i] = id_to_grp[id]
    return labels, groups
