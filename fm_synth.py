"""
Generate sound samples using fm synthesis
See http://www.cs.cmu.edu/~music/icm-online/readings/fm-synthesis/index.html

"""
import numpy as np


class FMSynthesizer(object):
    """

    """
    def __init__(self, rate=44100.0, carrier_init=(220.0, 0.5), modulation_init=(1.0, 0.0)):
        self._rate = rate
        self._carrier_freq, self._carrier_amp = carrier_init
        self._modulation_freq, self._modulation_depth = modulation_init
        self._t_offset = 0  # keep track of time offset to avoid pops when generating successive buffers

    def __str__(self):
        return "FMSynth(C_f = %.3f Hz, C_a = %.2f %%, M_f = %.3f Hz. M_d = %.2f)" % (self._carrier_freq,
                                                                                     self._carrier_amp * 100,
                                                                                     self._modulation_freq,
                                                                                     self._modulation_depth)

    def set_params(self, c_freq=None, c_amp=None, m_freq=None, m_depth=None):
        self._carrier_freq = c_freq if c_freq is not None else self._carrier_freq
        self._carrier_amp = c_amp if c_amp is not None else self._carrier_amp
        self._modulation_freq = m_freq if m_freq is not None else self._modulation_freq
        self._modulation_depth = m_depth if m_depth is not None else self._modulation_depth

    def get_samples(self, n, t_0=0.0):
        """
        Get the next N samples and the time of sample n+1
        """
        total_time = n / self._rate
        t = np.linspace(0, total_time, n + 1)[1:] + t_0

        modulation = np.sin(np.pi * 2.0 * self._modulation_freq * t) * self._modulation_depth
        samples = np.sin(np.pi * 2.0 * self._carrier_freq * t + modulation) * self._carrier_amp
        return samples, t[-1]

    def get_next_samples(self, n_samples):
        samples, self._t_offset = self.get_samples(n_samples, self._t_offset)
