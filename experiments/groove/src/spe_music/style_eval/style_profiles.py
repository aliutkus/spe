#!/usr/bin/env python3
import logging
from typing import Iterable

import muspy
import numpy as np
from confugue import Configuration, configurable

from . import note_features

_LOGGER = logging.getLogger(__name__)


def time_pitch_diff_hist(data: Iterable[muspy.Music],
                         max_time=2, bin_size=1/6, pitch_range=20, normed=True,
                         allow_empty=True):
    """Compute an onset-time-difference vs. interval histogram.

    Args:
        data: A list of MusPy `Music` objects.
        max_time: The maximum time between two notes to be considered.
        bin_size: The bin size along the time axis.
        pitch_range: The number of pitch difference bins in each direction (positive or negative,
            excluding 0). Each bin has size 1.
        normed: Whether to normalize the histogram.
        allow_empty: If `True`, a histogram will be computed even if there are no notes in the
            input. Otherwise, `None` will be returned in such a case.

    Returns:
        A 2D `np.array` of shape `[max_time / bin_size, 2 * pitch_range + 1]`, or `None` if
        `data` is empty.
    """
    epsilon = 1e-9
    time_diffs, intervals = [], []
    for music in data:
        notes = [n for tr in music.tracks for n in tr.notes]
        onsets = [n.start / music.resolution for n in notes]
        diff_mat = np.subtract.outer(onsets, onsets)

        # Count only positive time differences.
        index_pairs = zip(*np.where((diff_mat < max_time - epsilon) & (diff_mat >= 0.)))
        for j, i in index_pairs:
            if j == i:
                continue

            time_diffs.append(diff_mat[j, i])
            intervals.append(notes[j].pitch - notes[i].pitch)

    if not time_diffs and not allow_empty:
        return None

    with np.errstate(divide='ignore', invalid='ignore'):
        histogram, _, _ = np.histogram2d(
            intervals, time_diffs, normed=normed,
            bins=[np.arange(-(pitch_range + 1), pitch_range + 1) + 0.5,
                  np.arange(0., max_time + bin_size - epsilon, bin_size)])
    np.nan_to_num(histogram, copy=False)

    return histogram


NOTE_FEATURE_DEFS = {
    'duration': (note_features.Duration, {}),
    'onset': (note_features.OnsetPositionInBar, {}),
    'velocity': (note_features.Velocity, {}),
    'pitch': (note_features.Pitch, {})
}


NOTE_STAT_DEFS = [
    {
        'name': stat_name,
        'features': [{'name': feat_name} for feat_name in stat_name.split('.')],
    }
    for stat_name in ['onset', 'onset.duration', 'onset.velocity', 'pitch', 'onset.pitch']
]


@configurable
def extract_note_stats(data, *, _cfg):
    features = {key: _cfg['features'][key].configure(feat_type, **kwargs)
                for key, (feat_type, kwargs) in NOTE_FEATURE_DEFS.items()}
    feature_values = note_features.extract_features(data, features)

    @configurable
    def make_hist(name, normed=True, *, _cfg):
        feature_names = [f['name'] for f in _cfg.get('features')]
        with np.errstate(divide='ignore', invalid='ignore'):
            hist, _ = np.histogramdd(
                sample=[feature_values[name] for name in feature_names],
                bins=[_cfg['features'][i]['bins'].configure(features[name].get_bins)
                      for i, name in enumerate(feature_names)],
                normed=normed)
        np.nan_to_num(hist, copy=False)

        return name, hist

    # Create a dictionary mapping stat names to their values
    stats_cfg = _cfg['stats'] if 'stats' in _cfg else Configuration(NOTE_STAT_DEFS)
    return dict(stats_cfg.configure_list(make_hist))


@configurable
def extract_all_stats(data, *, _cfg):
    results = {}
    results['time_pitch_diff'] = _cfg['time_pitch_diff'].configure(
        time_pitch_diff_hist,
        data=data,
        normed=True,
        allow_empty=False)
    results.update(_cfg['note_stats'].configure(extract_note_stats, data=data))

    return {k: v for k, v in results.items() if v is not None}

