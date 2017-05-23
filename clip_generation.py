from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from activitynet import ActivityNet
import dataset
from utilities import dense_video_sampling


def main(dataset_name, dir_metadata, **kwargs):
    """Create list of clips and its annotations."""
    if dataset_name == 'activitynet':
        dset = ActivityNet(dir_metadata)
        ds_subsets = ['train', 'val']
    else:
        dset_class = getattr(dataset, dataset_name)
        dset = dset_class(dir_metadata)
        ds_subsets = dset.subsets

    for subset in ds_subsets:
        videos = dset.video_info(subset)
        annotations = dset.segments_info(subset)
        clips = dense_video_sampling(videos, annotations, **kwargs)
        filename = subset + '.lst'
        clips.to_csv(filename, sep=' ', header=None, index=None)


if __name__ == '__main__':
    description = 'Create list used by C3D binaries'
    p = ArgumentParser(description=description,
                       formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument('-ds', '--dataset-name', default='activitynet',
                   choices=['activitynet', 'Thumos14'])
    p.add_argument('-d', '--dir-metadata', required=True,
                   help='Root folder of dataset containing medatada folder')
    p.add_argument('-w', '--t-res', default=16, type=int,
                   help='temporal length of the clips')
    p.add_argument('-s', '--t-stride', default=16, type=int,
                   help='temporal stride used to extract clips')
    p.add_argument('-bl', '--bckg-label', default=200, type=int,
                   help='Integer label for background instances')

    main(**vars(p.parse_args()))
