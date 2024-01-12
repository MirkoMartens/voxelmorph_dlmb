import utils
import generators

# [['"data/test01.nii.gz"', '"data/test02.nii.gz"'], ['"data/test03.nii.gz"', '"data/test04.nii.gz"']]
train_files = utils.read_pair_list("C:/Users/mirko/Documents/Studium/Master/Semester05/DLMB/voxelmorph_dlmb/voxelmorph/py/test_input.txt")
print(train_files)
generator = generators.scan_to_scan(
        train_files, batch_size=1, bidir=True, add_feat_axis=False)
print("Done")
