import json
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
import pandas as pd
from pandas.io.json import json_normalize


class ActivityNet(object):
    """ActivityNet abstraction."""

    _partition_names = ['training', 'validation', 'testing']
    _partition_list = [('train_list.tsv', 'train'),
                       ('val_list.tsv', 'val'),
                       ('test_list.tsv', 'test')]
    _partition_segments_list = [('train_segments_list.tsv', 'train'),
                                ('val_segments_list.tsv', 'val'),
                                ('test_segments_list.tsv', 'test')]
    _extra_info = ['frame-rate', 'num-frames']

    def __init__(self, metadata_dir='non-existent',
                 annotation_file='non-existent.json',
                 extra_file='non-existent-file.tsv'):
        """Initialize ActivityNet dataset.

        Parameters
        ----------
        metadata_dir : str
            Fullpath of folder with ActivityNet metadata. Provide a location in
            the filesystem to allocate the metadata, if it does not exist.
        annotation_file : str
            Filename JSON-file with ground-truth annotation
        extra_file : str, optional
            Filename TSV-files (tab-separated values) with extra info about
            ActivityNet videos. Only 'video-name', 'frame-rate' and
            'num-frames' fields are taken in consideration

        """
        self.metadata = metadata_dir
        self.annotation_filename = annotation_file

        self.info_filename = None
        if os.path.isfile(extra_file):
            self.info_filename = extra_file

        if not os.path.isdir(self.metadata):
            os.makedirs(self.metadata)

        # Generate metadata about dataset, videos and segments
        self._base_df = None
        # Index for activitynet categories
        self.index_filename = os.path.join(self.metadata,
                                           'class_index_detection.tsv')

        # Video CSV
        self.files_video_list = [(os.path.join(self.metadata, i), j)
                                 for i, j in self._partition_list]
        for partition_file, subset in self.files_video_list:
            if not os.path.isfile(partition_file):
                self._dump_video_list(partition_file, subset)

        # Segments CSV
        self.files_seg_list = [(os.path.join(self.metadata, i), j)
                               for i, j in self._partition_segments_list]
        for segment_file, subset in self.files_seg_list:
            if not os.path.isfile(segment_file):
                self._dump_segments_info(segment_file, subset)

    def _dump_category_index(self, labels):
        """Write TSV-file with map between indexes and category labels.

        Parameters
        ----------
        labels : list
            List of string with category labels

        """
        labels.sort()
        df = pd.DataFrame(labels, columns=['activity-label'])
        df.to_csv(self.index_filename, sep='\t', index_label='idx-label')

    def _dump_video_list(self, filename, partition='train'):
        """Create TSV-file with information about ActivityNet videos.

        Parameters
        ----------
        filename : str
            Fullpath of TSV-file to save
        partition : str, optional
            ('train','val' or 'test') dump annotations of the corresponding set

        """
        # Make sure that ActivityNet ground-truth was parsed
        self._read_activitynet_json()

        # Focus on subset of interest
        subset = self._partition_names[self._partition_to_idx(partition)]
        idx_subset = self._base_df['subset'] == subset

        retained_keys = ['video-name', 'duration'] + self._retain_extra_info()
        df = self._base_df.loc[idx_subset, retained_keys]
        df.drop_duplicates(inplace=True)
        df.to_csv(filename, sep='\t', index=None)

    def _dump_segments_info(self, filename, partition):
        """Create TSV-file with data about ActivityNet activity segments.

        Parameters
        ----------
        filename : str
            Fullpath of TSV-file to save
        partition : str
            ('train','val' or 'test') dump annotations of the corresponding set

        """
        # Make sure that ActivityNet ground-truth was parsed
        self._read_activitynet_json()

        # Focus on subset of interest
        subset = self._partition_names[self._partition_to_idx(partition)]
        idx_subset = self._base_df['subset'] == subset
        retained_keys = ['video-name', 'duration', 't-init', 't-end',
                         'idx-label'] + self._retain_extra_info()
        if 'frame-rate' in retained_keys:
            retained_keys += ['f-init', 'f-end']
        df = self._base_df.loc[idx_subset, retained_keys]
        df.to_csv(filename, sep='\t', index=None)

    def _grab_extra_info(self, video_names):
        """Read TSV-file with additional data about videos.

        Parameters
        ----------
        video_names : list, ndarray, DataFrame, Series
            List of video name to query in info file

        Returns
        -------
        extra : dict
            Dict with extra information. Each value have the same size of
            video_names.

        """
        extra = {}
        if self.info_filename:
            df = pd.read_table(self.info_filename, index_col='video-name')
            for i in self._extra_info:
                if i not in df.columns:
                    continue
                extra[i] = df.loc[video_names, i].reset_index(drop=True)
        return extra

    def _partition_to_idx(self, partition):
        """Map partition to unique integer identifier.

        Parameters
        ----------
        partition : str
            Identifier for partition

        Returns
        -------
        idx : int
            integer identifier

        Raises
        ------
        ValueError
            Unrecognized partition

        """
        partition = partition.lower()
        if partition in ['train', 'training', 'trn']:
            return 0
        elif partition in ['val', 'validation']:
            return 1
        elif partition in ['test', 'testing', 'tst']:
            return 2
        else:
            raise ValueError('unrecognized choice')

    def _read_activitynet_json(self, id_prepend='v_'):
        """Parse JSON-file from ActivityNet.

        Parameters
        ----------
        id_prepend : str
            String to prepend to video-ids.

        """
        if isinstance(self._base_df, pd.DataFrame):
            return None

        with open(self.annotation_filename, 'r') as fobj:
            data = json.load(fobj)['database']

        # Use a list of dict instead of dict of dict
        video_id_list = data.keys()
        data_f = [None] * len(video_id_list)
        for i, video_id in enumerate(video_id_list):
            data_f[i] = data[video_id]
            data_f[i]['video-name'] = id_prepend + video_id
        # Skip annotaions
        keys = list(data_f[0])
        keys.remove('annotations')
        # JSON to Table using annotations as individaul entries
        gt_ = json_normalize(data_f, 'annotations', keys)

        # Dump activity label indexes
        if not os.path.isfile(self.index_filename):
            self._dump_category_index(gt_['label'].unique().tolist())

        # Grab t-init, t-end from gt_
        segments = np.array(gt_.loc[:, 'segment'].tolist())
        t_init, t_end = segments[:, 0], segments[:, 1]
        # Create index-label for each annotation segment
        idx_label = self.label_to_index(gt_.loc[:, 'label'])
        # Grab extra info (video-frames, frame-rate)
        extra = self._grab_extra_info(gt_.loc[:, 'video-name'])

        # Add additional columns to gt_
        extra['idx-label'] = idx_label
        extra['t-init'], extra['t-end'] = t_init, t_end
        if 'frame-rate' in extra:
            f_init = (extra['frame-rate'] * t_init).astype(int)
            f_end = (extra['frame-rate'] * t_end).astype(int)
            extra['f-init'], extra['f-end'] = f_init, f_end
        self._base_df = gt_.assign(**extra)

    def _retain_extra_info(self):
        """Return list of extra-info keys present in ground truth data.

        Returns
        -------
        list
            List of columns about extra info to preserve

        """
        return [i for i in self._extra_info if i in self._base_df.columns]

    def remap_annotations(self, fps):
        """Remap segments annotations.

        This function will edit [subset]_segments_list.tsv and
        [subset]_list.tsv files such that time and frame annotations are
        adjusted to a desired fps.

        Parameters
        ----------
        fps : int
            Desired frame rate

        Raises
        ------
        ValueError
            original frame-rate is not unknown

        """
        for filename, subset in self._partition_segments_list:
            df = self.segments_info(subset)
            if 'frame-rate' not in df.columns:
                raise ValueError(('Impossible to map annotations. '
                                  'Update TSV-files with frame-rate info'))

            df.loc[:, 't-init'] = (df.loc[:, 't-init'] * fps /
                                   df.loc[:, 'frame-rate'])
            df.loc[:, 't-end'] = (df.loc[:, 't-end'] * fps /
                                  df.loc[:, 'frame-rate'])
            df.loc[:, 'f-init'] = (fps * df.loc[:, 't-init']).astype(int)
            df.loc[:, 'f-end'] = (fps * df.loc[:, 't-end']).astype(int)

            df.to_csv(os.path.join(self.metadata, filename),
                      sep='\t', index=None)

        self.update_frame_rate(fps)

    def label_to_index(self, arr, reset_index=True):
        """Return index(es) of the activity label(s) in arr.

        Parameters
        ----------
        arr : List
            List of activity labels to query. It can handle all the data types
            supported by loc
        reset_index : bool, optional
            Reset index of Series

        Returns
        -------
        output: pandas.Series
            List of idx-label associated with the activity label

        Raises
        ------
        KeyError
            items (activity-label queries) are not found

        """
        df = pd.read_table(self.index_filename)
        df.set_index('activity-label', inplace=True)
        indexes = df.loc[arr, 'idx-label']
        if reset_index:
            indexes.reset_index(drop=True, inplace=True)
        return indexes

    def segments_info(self, partition='train'):
        """Return a DataFrame with information about action segments.

        Parameters
        ----------
        set_choice : string, optional
            ('train','val' or 'test') dump annotations of the corresponding
            set.
        filename : string, optional
            Fullpath to TSV-file with segments information.

        Returns
        -------
        df : pandas.DataFrame
            Table with info about every activity instance.

        """
        filename = self.files_seg_list[self._partition_to_idx(partition)][0]
        df = pd.read_table(filename)
        return df

    def update_frame_rate(self, fps):
        """Update frame rate field.

        Update [subset]_list.tsv and [subset]_segments_list.tsv files.

        Parameters
        ----------
        fps : int
            New frame rate

        """
        n_list = len(self._segments_list)
        for i, attribute in enumerate(self._partition_segments_list +
                                      self._segments_list):
            filename, subset = attribute
            if i < n_list:
                df = self.segments_info(subset)
            else:
                df = self.video_info(subset)

            n = len(df)
            df.loc[:, 'frame-rate'] = fps * np.ones(n)

            df.to_csv(filename, sep='\t', index=None)

    def update_num_frames(self, filename):
        """Update num-frames field.

         Update [subset]_list.tsv and [subset]_segments_list.tsv files.

        Parameters
        ----------
        filename : str
            Filename of TSV file with info (video-name, num-frames) fields.

        """
        new_df = pd.read_table(filename, index_col='video-name')
        n_list = len(self.files_video_list)
        for i, attribute in enumerate(self.files_video_list +
                                      self.files_seg_list):
            filename, subset = attribute
            if i >= n_list:
                df = self.segments_info(subset)
            else:
                df = self.video_info(subset)

            df.loc[:, 'num-frames'] = new_df.loc[df['video-name'],
                                                 'num-frames'].values

            df.to_csv(filename, sep='\t', index=None)

    def video_info(self, partition='train'):
        """Return DataFrame with info about videos on the corresponding set.

        Parameters
        ----------
        set_choice : string
            ('train', 'val' or 'test') set of interest

        Returns
        -------
        df : pandas.DataFrame
            Table with info about every video.

        """
        filename = self.files_video_list[self._partition_to_idx(partition)][0]
        df = pd.read_table(filename)
        return df


if __name__ == '__main__':
    description = ('Generate metadata about videos and segments, in the form '
                   'of TSV-files, from ActivityNet JSON-file annotation')
    p = ArgumentParser(description=description,
                       formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument('-d', '--metadata-dir', required=True,
                   help='Dirname of ActivityNet metadata folder')
    p.add_argument('-f', '--filename', required=True,
                   help='Filename of ActivityNet JSON-file annotations')
    p.add_argument('-e', '--extra-info-filename', default='none.tsv',
                   help='Filename of ActivityNet TSV-file with extra info')
    p.add_argument('-r', '--remap-fps', default=None, type=int,
                   help='Re-map instance annotations into specific FPS')
    args = p.parse_args()

    # Instanciate class to create metadata ;)
    dummy = ActivityNet(metadata_dir=args.metadata_dir,
                        annotation_file=args.filename,
                        extra_file=args.extra_info_filename)

    if args.remap_fps:
        print('Annotations have been re-mapped into different FPS')
        dummy.remap_annotations(args.remap_fps)
