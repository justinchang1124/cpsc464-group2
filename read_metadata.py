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
n = len(ids)

group_mapping = ['N/A', 'white', 'black', 'asian', 'native', 'hispanic', 'multi', 'hawa', 'amer indian']
id_to_grp = {}
id_to_lab = {}

for i in range(n):
    id_to_grp[ids[i]] = group_mapping[groups[i]]
    id_to_lab[ids[i]] = labels[i]

def get_labels_groups(t_ids):
    n = len(t_ids)
    t_labels = [None] * n
    t_groups = [None] * n
    for i in range(n):
        t_id = t_ids[i]
        t_labels[i] = id_to_lab[t_id]
        t_groups[i] = id_to_grp[t_id]
    return t_labels, t_groups
