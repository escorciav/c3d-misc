from collections import OrderedDict

import numpy as np
import pandas as pd


def dense_video_sampling(videos, annotations=None, bckg_label=201, t_res=16,
                         t_stride=16, drop_video=True):
    """Sample clips to extract C3D.

    Parameters
    ----------
    videos : pandas.DataFrame
        Table with info about videos in dataset i.e. unique entry per video.
        Required columns are video-name, num-frames.
    annotations : pandas.DataFrame, optional
        Used to set label of clips sampled from videos-table.
        Table with info about the annotations in dataset i.e. multiple videos.
        Required columns are video-name, idx-label, f-init, f-end.
    bckg_label : int
        Integer for background instances.
    t_res : int
        Temporal resolution of clips. It is given in terms of number of frames.
    t_stride : int
        Temporal resolution used to sample clips. It is given in terms of
        number of frames.
    drop_video : bool
        Drop video if it does not have annotations. This is different to set
        `annotations` to None.

    Returns
    -------
    df : pandas.DataFrame
        Table with info () about the clip.

    """
    clips = []
    for i, row in enumerate(videos.iterrows()):
        index, video = row
        video_name, num_frames = video['video-name'], video['num-frames']
        f_j = np.arange(1, num_frames - t_res + 1, t_stride, dtype=int)
        num_clips = len(f_j)

        if annotations is None:
            index_labels = bckg_label * np.ones(num_clips, dtype=int)
        else:
            idx = annotations['video-name'] == video_name
            if idx.sum() == 0:
                continue
            targets = annotations.loc[idx, ['f-init', 'f-end',
                                            'idx-label']].values
            segments = np.empty((num_clips, 2), dtype=int)
            segments[:, 0] = f_j
            segments[:, 1] = f_j + t_res - 1

            # Assign label to clips with overlap >= t_res/2 to instances
            overlap = intersection_area(targets[:, 0:2], segments)
            idx_target = np.argmax(overlap, axis=0)
            index_labels = targets[idx_target, 2]
            index_labels[overlap.max(axis=0) < t_res/2] = bckg_label

        clips.append(
            pd.DataFrame(
                OrderedDict([('video-name', [video_name]*num_clips),
                             ('f-init', f_j),
                             ('idx-label', index_labels)])))

    clips_df = pd.concat(clips, ignore_index=True, copy=False)
    return clips_df


def intersection_area(target_segments, test_segments):
    """Compute area/length of overlap btw segments.

    Parameters
    ----------
    target_segments : ndarray.
        2d-ndarray of size [m, 2] with format [t-init, t-end].
    test_segments : ndarray.
        2d-ndarray of size [n x 2] with format [t-init, t-end].

    Outputs
    -------
    overlap : ndarray
        2d-ndarray of size [m x n] with overlap.

    Raises
    ------
    ValueError
        target_segments or test_segments are not 2d-ndarray.

    Notes
    -----
    It assumes that target-segments are more scarce that test-segments

    """
    if target_segments.ndim != 2 or test_segments.ndim != 2:
        raise ValueError('Dimension of arguments is incorrect')

    m, n = target_segments.shape[0], test_segments.shape[0]
    overlap = np.empty((m, n))
    for i in range(m):
        tt1 = np.maximum(target_segments[i, 0], test_segments[:, 0])
        tt2 = np.minimum(target_segments[i, 1], test_segments[:, 1])

        overlap[i, :] = (tt2 - tt1 + 1.0).clip(0)
    return overlap


def iou(target_segments, test_segments):
    """Compute intersection over union btw segments.

    Parameters
    ----------
    target_segments : ndarray.
        2d-ndarray of size [m, 2] with format [t-init, t-end].
    test_segments : ndarray.
        2d-ndarray of size [n x 2] with format [t-init, t-end].

    Outputs
    -------
    iou : ndarray
        2d-ndarray of size [m x n] with tIoU ratio.

    Raises
    ------
    ValueError
        target_segments or test_segments are not 2d-ndarray.

    Notes
    -----
    It assumes that target-segments are more scarce that test-segments

    """
    if target_segments.ndim != 2 or test_segments.ndim != 2:
        raise ValueError('Dimension of arguments is incorrect')

    m, n = target_segments.shape[0], test_segments.shape[0]
    iou = np.empty((m, n))
    for i in range(m):
        tt1 = np.maximum(target_segments[i, 0], test_segments[:, 0])
        tt2 = np.minimum(target_segments[i, 1], test_segments[:, 1])

        # Non-negative overlap score
        intersection = (tt2 - tt1 + 1.0).clip(0)
        union = ((test_segments[:, 1] - test_segments[:, 0] + 1) +
                 (target_segments[i, 1] - target_segments[i, 0] + 1) -
                 intersection)
        # Compute overlap as the ratio of the intersection
        # over union of two segments at the frame level.
        iou[i, :] = intersection / union
    return iou
