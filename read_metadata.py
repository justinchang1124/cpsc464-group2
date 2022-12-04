import numpy as np

abs_proj_path = 'C:/Users/justin/PycharmProjects/cpsc464-group2'
groups_path = 'resources/groups.txt'
labels_path = 'resources/labels.txt'
ids_path = 'resources/patient_ids.txt'
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


def id_to_label(id):
    return min(max(id_to_lab[id], 0), 4)  # clamp to [0, 1, 2, 3, 4]


def id_to_group(id):
    return id_to_grp[id]


def empty_group_counter():
    result = {}
    for group in group_mapping:
        result[group] = 0
    return result
