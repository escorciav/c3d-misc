import array
import glob
import os
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import h5py
import numpy as np


def read_feature(filename, keep_shape=False):
    """Read feature (a.k.a blob) dump by C3D.

    Parameters
    ----------
    filename : str
        Fullpath of file to read.
    keep_shape : bool
        Reshape feature to the shape reported.

    Outputs
    -------
    feature : ndarray
        numpy array of features
    s : tuple
        shape of original feature

    Note: It accomplishes the same purpose of this code:
        C3D/examples/c3d_feature_extraction/script/read_binary_blob.m

    """
    s_parr, d_parr = array.array('i'), array.array('f')
    with open(filename, 'rb') as f:
        s_parr.fromfile(f, 5)
        s = np.array(s_parr)
        m = np.cumprod(s)[-1]

        d_parr.fromfile(f, m)

    feature = np.array(d_parr)
    if keep_shape:
        feature = feature.reshape(s)
    return feature, s


def read_all_features_video(dirname, layer, dtype=np.float32, keep_shape=True):
    """Stack all the blobs from inside a folder into a numpy array"""
    if not os.path.exists(dirname):
        raise IOError('Unexistent folder {}'.format(dirname))

    c3d_files = glob.glob(os.path.join(dirname, '*' + layer))
    if len(c3d_files) == 0:
        print('No files to read for: {}'.format(os.path.basename(dirname)))
        return
    sorted_files = sorted(c3d_files)
    # Initialize ndarray
    data, s = read_feature(sorted_files[0], keep_shape)
    s[0] = len(sorted_files)
    arr = np.empty(tuple(s), dtype=dtype)
    arr[0, ...] = data

    # Read features
    for i, v in enumerate(sorted_files[1::]):
        data, _ = read_feature(v, keep_shape)
        arr[i + 1, ...] = data
    return arr


def main(root_dir, output_file, layers=['fc6-1'], hdf5_mode='w',
         freq_interval=10):
    """Save C3D-blob binaries as HDF5.

    It recursively save all the blobs from one layer inside a root folder
    into an HDF5. It creates a GROUP for each subfolder inside the root and
    stores the blob into a DATASET name c3d_{layer}.

    """
    compression_flags = dict(compression="gzip", compression_opts=9)
    with h5py.File(output_file, hdf5_mode) as f:
        video_names = os.listdir(root_dir)
        n_videos = len(video_names)
        cum_time = 0
        for i, video_it in enumerate(video_names):
            start_time = time.time()
            for l in layers:
                arr = read_all_features_video(
                    os.path.join(root_dir, video_it), l, dtype=np.float32)
                if arr.size > 0:
                    if video_it not in f:
                        g = f.create_group(video_it)
                    else:
                        g = f[video_it]
                    g.create_dataset('c3d_{}'.format(l), data=arr,
                                     chunks=True, **compression_flags)
            iter_time = time.time() - start_time
            cum_time += iter_time
            if (i + 1) % freq_interval == 0:
                msg = 'Iter: {}/{}\tElapsed time: {}'
                print(msg.format(i + 1, n_videos, cum_time))


if __name__ == '__main__':
    description = 'Save C3D features as HDF5'
    p = ArgumentParser(description=description,
                       formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument('-r', '--root-dir', required=True,
                   help='Dirname of root allocation features per video')
    p.add_argument('-o', '--output-file', required=True,
                   help='Name of hdf5 file to create')
    p.add_argument('-l', '--layers', nargs='+', default=['fc6-1'],
                   help='layer extracted which corresponds to file extension')
    p.add_argument('-h5m', '--hdf5_mode', default='w',
                   help='Mode used to open HDF5 output file')
    p.add_argument('-fqi', '--freq-interval', type=int, default=20,
                   help='Frequency interval to write progress')

    main(**vars(p.parse_args()))
