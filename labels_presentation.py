import numpy as np


# dict.get() but return [] if not found
def key_get_list(examp_dict, examp_key):
    result = examp_dict.get(examp_key)
    if result is None:
        return []
    return result


# separates z_pred, z_true by grp_keys
def separate_by_group(z_pred, z_true, grp_keys):
    n_pred = len(z_pred)
    if n_pred != len(z_true):
        raise ValueError("Length of predictions does not equal length of truths!")
    if n_pred != len(grp_keys):
        raise ValueError("Length of predictions does not equal number of keys!")

    a_dict = {}  # a_dict[group] is the actual predictions within that group
    r_dict = {}  # r_dict[group] is the real values within that group

    for i in range(n_pred):
        grp_key = grp_keys[i]
        a_list = key_get_list(a_dict, grp_key)
        r_list = key_get_list(r_dict, grp_key)

        a_list.append(z_pred[i])
        r_list.append(z_true[i])

        a_dict[grp_key] = a_list
        r_dict[grp_key] = r_list

    return a_dict, r_dict


# summarizes the result of separate_by_group
def summarize_ar_dict(a_dict, r_dict):
    for grp_key in a_dict.keys():
        r_list = r_dict[grp_key]
        a_list = a_dict[grp_key]
        d_list = [a_i - b_i for a_i, b_i in zip(a_list, r_list)]  # differences list
        n_sup = 0
        n_sub = 0
        n_cor = 0
        for diff in d_list:
            if diff == 0:
                n_cor += 1
            elif diff > 0:
                n_sup += 1
            else:
                n_sub += 1

        total = len(d_list)
        print("{} accuracy rate: {}=, {}+, {}-, total {}/{}".format(grp_key, n_cor, n_sup, n_sub, n_cor / total))
        avg_re = sum(r_list) / total
        avg_ai = sum(a_list) / total
        print("{} average real vs AI: {} vs {}".format(grp_key, avg_re, avg_ai))
        df_var = np.var(d_list)
        df_mad = sum(map(abs, d_list)) / total
        print("{} difference: Variance = {}, Mean Absolute Difference = {}".format(grp_key, df_var, df_mad))


# example usage
# ea_dict, er_dict = separate_by_group([1, 2, 1, 2, 3, 4], [1, 2, 1, 2, 1, 2], ['a', 'a', 'b', 'b', 'c', 'c'])
# print(ea_dict, er_dict)
# summarize_ar_dict(ea_dict, er_dict)