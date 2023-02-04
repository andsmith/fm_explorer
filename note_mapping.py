"""
Classes to map from y-position on staff to frequency.
Implement temperaments, keys, auto-tune, etc.
"""

from scipy.interpolate import interp1d
import numpy as np
from abc import ABCMeta, abstractmethod

MIDDLE_C_HZ = 261.625565


class NoteMap(metaclass=ABCMeta):
    """
    Abstract class to map from space on the staff to frequency.
    """

    def __init__(self, staff_middle_y, staff_space, middle_c_hz=None, octave_range=(-10, 10)):
        """
        :param staff_middle_y:  y-coordinate of middle of staff (middle-c)
        :param staff_space:  distance between ledger lines
        :param middle_c_hz: Frequency of staff_middle_y
        :param octave_range: clip to this range around middle C
        """

        self._m_y = staff_middle_y
        self._space = staff_space / 4  # space between A and A# on staff
        m_c = middle_c_hz if middle_c_hz is not None else MIDDLE_C_HZ
        self._octave_range = octave_range
        self._c_bottom = m_c * 2. ** octave_range[0]
        self._y_bottom = self._m_y + self._space * 14 * (-octave_range[0])

        self._freqs, self._note_y_pos = self._get_freqs_and_locations()

        self._note_y_div = (self._note_y_pos[:-1] + self._note_y_pos[1:]) / 2  # halfway between notes (bin boundaries)
        self._gliss = interp1d(self._note_y_pos, self._freqs)
        self._middle_c_staff_distance = self._m_y - self._y_bottom

    def get_note(self, y):
        """
        What note is position Y on the staff?
        Find the closest two 'proper' notes and interpolate.
        :returns: freq_raw: frequency determined by mouse
                   freq: Closest chromatic pitch}
        """
        n = np.sum(y < self._note_y_div)
        freq_raw = self._gliss(y)
        freq = self._freqs[n]
        return freq_raw, freq

    @abstractmethod
    def _get_freqs_and_locations(self):
        """
        Determine what frequencies are allowed and where they go on the staff.
        :returns:  list of frequencies in increasing order,
                   list of y coordinates (bottom up) of corresponding locations
        """
        pass


class EqualNoteMap(NoteMap):
    """
    Equal Temperament map from pixel location on staff to frequency.
    Autotune, key signatures happen here.
    """
    # half steps from C:
    ALLOWED_NOTES = [i for i in range(12)]  # all allowed for chromatic scale

    def _get_freqs_and_locations(self):
        """
        Set list of allowed "note" frequencies, and their y-coordinate on the staff
        """
        oct_span = self._octave_range[1] - self._octave_range[0]
        n_notes = oct_span * 12

        freqs = self._c_bottom * 2 ** (np.arange(n_notes) / 12.)  # all valid frequencies
        note_y_pos = self._y_bottom - self._space * np.array([i for i in range(oct_span * 14)
                                                              if not (
                    i % 14 == 6 or i % 14 == 13)])  # skip E-F, B-C gap

        return self.__class__._filter(freqs, note_y_pos)

    @classmethod
    def _filter(cls, freqs, note_y_pos):
        allowed = [i for i in range(len(freqs)) if i % 12 in cls.ALLOWED_NOTES]
        return freqs[allowed], note_y_pos[allowed]


class DiatonicNoteMap(EqualNoteMap):
    """
    White keys
    """
    ALLOWED_NOTES = [0, 2, 4, 5, 7, 9, 11]  # half steps from C


class PentatonicNoteMap(DiatonicNoteMap):
    """
    Black keys
    """
    ALLOWED_NOTES = [i for i in range(12) if i not in DiatonicNoteMap.ALLOWED_NOTES]


def _test_note_maps():
    e = EqualNoteMap(500, 10)
    d = DiatonicNoteMap(500, 10)
    p = PentatonicNoteMap(500, 10)
    assert e._freqs.size > d._freqs.size > p._freqs.size, "Scales have wrong numbers of notes!"


if __name__ == "__main__":
    _test_note_maps()
    print("All tests pass.")
