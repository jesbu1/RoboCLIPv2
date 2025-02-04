import h5py
import numpy as np
import os

h5_file_name = "usc_koch_rewind_reward.h5"

h5_file = h5py.File(h5_file_name, 'r')
array = np.asarray(h5_file['Fold the blue towel']["2"])

array_0 = array[0]
array_1 = array[1]
diff_sum = np.sum(np.abs(array_0 - array_1))

first_col_std = np.std(array[:, 0])
sec_col_std = np.std(array[:, 1])
third_col_std = np.std(array[:, 2])
print("koch array diff", diff_sum)
print("koch array std colunm 1, 2, 3", first_col_std, sec_col_std, third_col_std)
import pdb ; pdb.set_trace()

h5_file_name = "jesse_collect_dataset_new_token.h5"
h5_file = h5py.File(h5_file_name, 'r')
array = np.asarray(h5_file['put the black bowl in the pot']["2"])[:32]
array_0 = array[0]
array_1 = array[1]
diff_sum = np.sum(np.abs(array_0 - array_1))
first_col_std = np.std(array[:, 0])
sec_col_std = np.std(array[:, 1])
third_col_std = np.std(array[:, 2])

print("jesse array diff", diff_sum)
print("jesse array std colunm 1, 2, 3", first_col_std, sec_col_std, third_col_std)
