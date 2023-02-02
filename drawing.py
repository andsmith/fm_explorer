import numpy as np
import time
import cv2
from gui_utils.drawing import draw_rect, draw_box
from scipy.fft import fft  # fastest
from numpy.fft import fftfreq
from abc import abstractmethod, ABCMeta


class DataAnimation(metaclass=ABCMeta):
    """
    Generic live display of data
    """

    def __init__(self, bbox, color=(255, 255, 255, 255)):
        """
        :param bbox: dict with 'top','left','right','bottom', where in image to draw animation
        """
        self._bbox = bbox
        self._color = color
        self._coords = None
        self._last_params = None
        self._last_coords = None
        self._precision_bits = 6
        self._precision_mult = 2 ** self._precision_bits
        self._bbox_height = self._bbox['bottom'] - self._bbox['top']
        self._bbox_width = self._bbox['right'] - self._bbox['left']

    @abstractmethod
    def set_samples(self, samples):
        """
        re-define current frame
        """
        pass

    def draw(self, image):
        """
        Render wave in image.  Params optional, used to avoid re-calculating coords.
        """
        cv2.polylines(image, [self._coords], False, self._color, 1, cv2.LINE_AA, self._precision_bits)


class AnimatedSpectrum(DataAnimation):

    def __init__(self, bbox, f_range, sampling_rate=44100, color=(255, 255, 255, 255)):
        """
        bbox-bounding box
        f_range (f_low, f_high) for display
        sampling_rate to get frequencies correct in FFT.
        """
        super(AnimatedSpectrum, self).__init__(bbox, color=color)
        self._f_range = f_range
        self._p_range = (0.0, 1.0)
        self._sampling_rate = sampling_rate
        self._samples = None
        self._p = None

    def set_f_range(self, f_range, p_range):
        self._f_range = f_range
        self._recalculate_coords()

    def set_samples(self, samples):
        self._samples = samples
        self._recalculate_spectrum()
        self._recalculate_coords()

    def _recalculate_coords(self):
        """
        Prune spectrum, scale to window
        """
        # first determine frequencies
        f_valid = np.logical_and(self._power_f >= self._f_range[0], self._power_f <= self._f_range[1])

        # apply cutoff and scale
        spectrum = self._log_power[f_valid]
        spectrum = spectrum - np.min(spectrum)
        spectrum_normalized = 1.0 - spectrum / np.max(spectrum)  # flip y
        y_values = spectrum_normalized * self._bbox_height + self._bbox['top']
        x_values_unit = (self._power_f[f_valid] - self._f_range[0]) / (self._f_range[1] - self._f_range[0])
        x_values = x_values_unit * self._bbox_width + self._bbox['left']

        self._coords = (self._precision_mult * np.dstack([x_values, y_values])).reshape(-1, 1, 2).astype(np.int32)

    def _recalculate_spectrum(self):
        """
        Determine shape of spectrum curve, from samples & Frequency bounds
        """
        # now get spectrum
        size = int(self._samples.size / 2)
        z = fft(self._samples)[:size]
        self._log_power = np.log10(np.abs(z) ** 2.)
        f = fftfreq(self._samples.size, 1. / self._sampling_rate)[:size]

        self._power_f = f
        self._peak_power = np.max(self._log_power)


class AnimatedWave(DataAnimation):
    def set_samples(self, samples):
        """
        Show waveform, just scale samples to bounding box in x-direction, from [-1,+1] to [0, 1] in Y.
        """
        y_values = (-samples / 2. + 0.5) * self._bbox_height + self._bbox['top']
        x_values = np.linspace(self._bbox['left'], self._bbox['right'] - 1, y_values.size)
        self._coords = (self._precision_mult * np.dstack([x_values, y_values])).reshape(-1, 1, 2).astype(np.int32)


def eval_ellipse(t, ecc, phi_rad, h):
    _, b = get_ellipse_axes(h, ecc, phi_rad)
    radius = b / np.sqrt(1. - (ecc * np.cos(t + phi_rad)) ** 2.)
    if isinstance(t, np.ndarray):
        return np.stack((radius * np.cos(t),
                         radius * np.sin(t)), axis=1)
    return np.array((radius * np.cos(t),
                     radius * np.sin(t))).reshape(1, 2)


def get_ellipse_axes(h, ecc, phi_rad):
    """
    Calculate the major/minor axes of an ellipse.
    from https://math.stackexchange.com/questions/1535774/height-of-a-rotated-ellipse
    :param h: half distance between staff lines
    :param ecc: eccentricity of notehead
    :param phi_rad: rotation of notehead in radians
    :returns: a, b, major/minor axis lengths
    """
    cos2phi = np.cos(2. * phi_rad)
    qd2 = (1. / (1. - ecc ** 2.)) * (1. - cos2phi) + cos2phi + 1
    b = h / np.sqrt(qd2 / 2.)
    a = np.sqrt(b ** 2. / (1. - ecc ** 2.))
    y = np.sqrt(0.5 * (a ** 2. + b ** 2. - (a ** 2. - b ** 2.) * np.cos(2. * phi_rad)))
    assert np.abs(y - h) < 1e-8, "Error calculating ellipse params."
    return a, b
