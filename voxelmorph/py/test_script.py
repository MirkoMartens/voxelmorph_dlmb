import utils
import voxelmorph as vxm
import os
import glob
import numpy as np
import sys

# [['"data/test01.nii.gz"', '"data/test02.nii.gz"'], ['"data/test03.nii.gz"', '"data/test04.nii.gz"']]
train_files = utils.read_pair_list("voxelmorph_dlmb/train_list.txt")
#print(train_files)


def load_volfile_pair(pair, **kwargs):
    """
    Load a pair of volume files.

    Parameters:
        pair: Tuple containing two file paths or two preloaded volumes.
        kwargs: Forwarded to load_volfile for each volume in the pair.

    Returns:
        Tuple of loaded volumes.
    """
    return tuple(utils.load_volfile(vol, **kwargs) for vol in pair)



def volgen(
    vol_names,
    batch_size=1,
    segs=None,
    np_var='vol',
    pad_shape=None,
    resize_factor=1,
    add_feat_axis=True
):
    """
    Base generator for random volume loading. Volumes can be passed as a path to
    the parent directory, a glob pattern, a list of file paths, or a list of
    preloaded volumes. Corresponding segmentations are additionally loaded if
    `segs` is provided as a list (of file paths or preloaded segmentations) or set
    to True. If `segs` is True, npz files with variable names 'vol' and 'seg' are
    expected. Passing in preloaded volumes (with optional preloaded segmentations)
    allows volumes preloaded in memory to be passed to a generator.

    Parameters:
        vol_names: Path, glob pattern, list of volume files to load, or list of
            preloaded volumes.
        batch_size: Batch size. Default is 1.
        segs: Loads corresponding segmentations. Default is None.
        np_var: Name of the volume variable if loading npz files. Default is 'vol'.
        pad_shape: Zero-pads loaded volumes to a given shape. Default is None.
        resize_factor: Volume resize factor. Default is 1.
        add_feat_axis: Load volume arrays with added feature axis. Default is True.
    """
    # convert glob path to filenames
    if isinstance(vol_names, str): # todo adjust this to make vol_names list of tuples
        if os.path.isdir(vol_names):
            vol_names = os.path.join(vol_names, '*')
        vol_names = glob.glob(vol_names)



    if isinstance(segs, list) and len(segs) != len(vol_names):
        raise ValueError('Number of image files must match number of seg files.')

    while True:
        # generate [batchsize] random image indices
        indices = np.random.randint(len(vol_names), size=batch_size)

        # load volumes and concatenate
        load_params = dict(np_var=np_var, add_batch_axis=True, add_feat_axis=add_feat_axis,
                           pad_shape=pad_shape, resize_factor=resize_factor)
        pairs = [load_volfile_pair(vol_names[i], **load_params) for i in indices] 
        
        imgs1, imgs2 = zip(*pairs)
        vol1 = np.concatenate(imgs1, axis=0)
        vol2 = np.concatenate(imgs2, axis=0)

        vols = [vol1, vol2]

        # optionally load segmentations and concatenate
        # if segs is True:
        #     # assume inputs are npz files with 'seg' key
        #     load_params['np_var'] = 'seg'  # be sure to load seg
        #     s = [py.utils.load_volfile(vol_names[i], **load_params) for i in indices]
        #     vols.append(np.concatenate(s, axis=0))
        # elif isinstance(segs, list):
        #     # assume segs is a corresponding list of files or preloaded volumes
        #     s = [py.utils.load_volfile(segs[i], **load_params) for i in indices]
        #     vols.append(np.concatenate(s, axis=0))

        yield tuple(vols)


# vol_names is a list of lists: [['"data/test01.nii.gz"', '"data/test02.nii.gz"'], ['"data/test03.nii.gz"', '"data/test04.nii.gz"']]
def scan_to_scan(vol_names, bidir=False, batch_size=1, prob_same=0, no_warp=False, **kwargs):
    """
    Generator for scan-to-scan registration.

    Parameters:
        vol_names: List of volume files to load, or list of preloaded volumes.
        bidir: Yield input image as output for bidirectional models. Default is False.
        batch_size: Batch size. Default is 1.
        prob_same: Induced probability that source and target inputs are the same. Default is 0.
        no_warp: Excludes null warp in output list if set to True (for affine training). 
            Default if False.
        kwargs: Forwarded to the internal volgen generator.
    """
    zeros = None
    gen = volgen(vol_names, batch_size=batch_size, **kwargs)
    while True:
        scan_pair = next(gen)
        scan1 = scan_pair[0]
        scan2 = scan_pair[1]

        # some induced chance of making source and target equal
        if prob_same > 0 and np.random.rand() < prob_same:
            if np.random.rand() > 0.5:
                scan1 = scan2
            else:
                scan2 = scan1

        # cache zeros
        if not no_warp and zeros is None:
            shape = scan1.shape[1:-1]
            zeros = np.zeros((batch_size, *shape, len(shape)))

        invols = [scan1, scan2]
        outvols = [scan2, scan1] if bidir else [scan2]
        if not no_warp:
            outvols.append(zeros)

        yield (invols, outvols)



generator = vxm.generators.scan_to_scan(
        train_files, batch_size=1, bidir=True, add_feat_axis=False)

output = next(generator)
#print("vol_names: "+train_files+"\n")
# convert glob path to filenames
if isinstance(train_files, str): # todo adjust this to make vol_names list of tuples
        if os.path.isdir(train_files):
                train_files = os.path.join(train_files, '*')
        train_files = glob.glob(train_files)

        
print("Done")
