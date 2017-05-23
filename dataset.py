import os
import glob

import pandas as pd


class VideoDataset(object):
    """Generic VideoDataset

    Attibutes:
        subsets (list) : list of subsets predefined for the dataset.
    """
    def __init__(self):
        self.subsets = ['all']
        pass

    def segments_info(self):
        raise NotImplemented('Overwrite this method')

    def video_info(self):
        raise NotImplemented('Overwrite this method')


class Thumos14(VideoDataset):
    """Pack data about Thumos14 dataset."""
    def __init__(self, root):
        """Initialize thumos14 class.

        Parameters
        ----------
        dirname : string
            Fullpath of folder with THUMOS-14 data

        """
        if not os.path.isdir(root):
            raise IOError('Unexistent directory {}'.format(root))
        self.root = root
        self.subsets = ['val', 'test']

        # Read index used on THUMOS-14
        filename = os.path.join(self.root, 'class_index_detection.txt')
        self.df_index_labels = pd.read_csv(filename, header=None, sep=' ')

        # Video CSV
        self.files_video_list = [
            os.path.join(self.root, 'metadata', 'val_list.txt'),
            os.path.join(self.root, 'metadata', 'test_list.txt')]
        msg = 'Unexistent list of {} videos and its information'
        # TODO: Generate list if not exist
        if not os.path.isfile(self.files_video_list[0]):
            raise IOError(msg.format('validation'))
        if not os.path.isfile(self.files_video_list[1]):
            raise IOError(msg.format('testing'))

        # Segments CSV
        self.files_seg_list = [
            os.path.join(self.root, 'metadata', 'val_segments_list.txt'),
            os.path.join(self.root, 'metadata', 'test_segments_list.txt')]
        if not os.path.isfile(self.files_seg_list[0]):
            raise NotImplementedError('implementation in progress')
            # TODO: self._gen_segments_info(self.files_seg_list[0], 'val')
        if not os.path.isfile(self.files_seg_list[1]):
            raise NotImplementedError('implementation in progress')
            # TODO: self._gen_segments_info(self.files_seg_list[1], 'test')

    def annotation_files(self, set_choice='val'):
        """Return files of temporal annotations of THUMOS-14 actions.

        Parameters
        ----------
        set_choice : string, optional
            ('val' or 'test') set of interest

        """
        dirname = self.dir_annotations(set_choice)
        return glob.glob(os.path.join(dirname, 'annotation', '*.txt'))

    def dir_annotations(self, set_choice='val'):
        """Return string of folder of annotations.

        Parameters
        ----------
        set_choice : string, optional
            ('val' or 'test') set of interest

        """
        set_choice = set_choice.lower()
        if set_choice == 'val' or set_choice == 'validation':
            return os.path.join(self.root, 'th14_temporal_annotations_val')
        elif (set_choice == 'test' or set_choice == 'testing' or
              set_choice == 'tst'):
            return os.path.join(self.root, 'th14_temporal_annotations_test')
        else:
            raise ValueError('unrecognized choice')

    def dir_videos(self, set_choice='val'):
        """Return string of folder with videos.

        Parameters
        ----------
        set_choice : string, optional
            ('val' or 'test') return folder of the corresponding set

        """
        set_choice = set_choice.lower()
        if set_choice == 'val' or set_choice == 'validation':
            return os.path.join(self.root, 'val_mp4')
        elif (set_choice == 'test' or set_choice == 'testing' or
              set_choice == 'tst'):
            return os.path.join(self.root, 'test_mp4')
        else:
            raise ValueError('unrecognized choice')

    def segments_info(self, set_choice='val', filename=None):
        """Return a DataFrame with information about THUMOS-14 action segments.

        Parameters
        ----------
        set_choice : string, optional
            ('val' or 'test') dump annotations of the corresponding set

        """
        set_choice = set_choice.lower()
        if set_choice in ['val', 'validation']:
            filename = self.files_seg_list[0]
        elif set_choice in ['test', 'testing', 'tst']:
            filename = self.files_seg_list[1]
        else:
            raise ValueError('unrecognized choice')

        df = pd.read_csv(filename, sep='\t')
        return df

    def video_info(self, set_choice='val'):
        """Return DataFrame with info about videos on the corresponding set.

        Parameters
        ----------
        set_choice : string
            ('val' or 'test') set of interest

        """
        set_choice = set_choice.lower()
        if set_choice in ['val', 'validation']:
            filename = self.files_video_list[0]
        elif set_choice in ['test', 'testing', 'tst']:
            filename = self.files_video_list[1]
        else:
            raise ValueError('unrecognized choice')

        df = pd.read_csv(filename, sep='\t')
        return df
