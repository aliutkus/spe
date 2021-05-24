from typing import Iterable
import muspy
import numpy as np
from note_seq import midi_io

_EPSILON = 1e-9


def extract_features(data: Iterable[muspy.Music], features):
    """Extract a set of features from the given note sequences.

    Args:
        note_sequences: an iterable of MusPy `Music` objects.
        features: a dictionary with feature objects as values.

    Returns:
        A dictionary mapping keys from `features` to lists of feature values.
    """
    results = {key: [] for key in features}
    for music in data:
        for key, feature in features.items():
            results[key].extend(list(feature.extract(music)))

    assert len(set(len(x) for x in results.values())) <= 1

    return results


class Pitch:
    """The MIDI pitch of the note."""

    def extract(self, music: muspy.Music):
        for track in music.tracks:
            for note in track.notes:
                yield note.pitch

    def get_bins(self, min_value=0, max_value=127):
        return np.arange(min_value, max_value + 1) - 0.5


class Duration:
    """The duration of the note.

    It is assumed that the tempo is normalized (typically to 60 BPM) so that the duration is
    expressed in beats.
    """

    def extract(self, music: muspy.Music):
        for track in music.tracks:
            for note in track.notes:
                yield note.duration / music.resolution

    def get_bins(self, bin_size=1/6, max_value=2):
        return np.arange(0., max_value + bin_size - _EPSILON, bin_size)


class Velocity:
    """The MIDI velocity of the note."""

    def extract(self, music: muspy.Music):
        for track in music.tracks:
            for note in track.notes:
                yield note.velocity

    def get_bins(self, num_bins=8):
        return np.arange(0, 127, 128 / num_bins) - 0.5


class OnsetPositionInBar:
    """The time of the note onset expressed in beats from the most recent downbeat."""

    def extract(self, music: muspy.Music):
        if music.time_signatures:
            if (len(set((ts.numerator, ts.denominator) for ts in music.time_signatures)) > 1
                    or music.time_signatures[0].time > 0):
                raise NotImplementedError('Music with multiple time signatures is not supported')
            bar_duration = music.time_signatures[0].numerator * music.resolution
            # TODO: Make sure this is correct and handle compound time signatures.
        else:
            # Assume 4/4
            bar_duration = 4 * music.resolution

        for track in music.tracks:
            for note in track.notes:
                yield note.start % bar_duration

    def get_bins(self, bin_size=1/6, max_beats=4):
        return np.arange(0., max_beats + bin_size - _EPSILON, bin_size)
