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
        self._params = {'carrier_freq': carrier_init[0],
                        'carrier_amp': carrier_init[1],
                        'mod_freq': modulation_init[0],
                        'mod_depth': modulation_init[0]}

        self._last_params = {p: None for p in self._params}  # for smooth interpolation

        self._t_offset = 0  # keep track of time offset to avoid pops when generating successive buffers

    def __str__(self):
        return "FMSynth(C_f = %.3f Hz, C_a = %.2f %%, M_f = %.3f Hz. M_d = %.2f)" % (self._params['carrier_freq'],
                                                                                     self._params['carrier_amp'] * 100,
                                                                                     self._params['mod_freq'],
                                                                                     self._params['mod_depth'])

    def set_param(self, name, value):
        """
        Remember the previous value for smooth interpolation
        TODO:   Keep ALL previous values, tag w/timestamp, so values changing too fast aren't lost.
        """
        self._last_params[name] = self._params[name]
        self._params[name] = value

    def get_samples(self, n, t_0=0.0, smoothed=False):
        """
        Get the next N samples and the time of sample n+1
        """
        total_time = n / self._rate
        t = np.linspace(0, total_time, n + 1)[1:] + t_0

        params = self._params.copy()
        if smoothed:
            for name in self._params:
                if self._last_params[name] is not None:
                    params[name] = np.linspace(self._last_params[name], self._params[name], n)
                    self._last_params[name] = None  # only do this once!

        modulation = np.sin(np.pi * 2.0 * params['mod_freq'] * t) * params['mod_depth']
        samples = np.sin(np.pi * 2.0 * params['carrier_freq'] * t + modulation) * params['carrier_amp']
        return samples, t[-1]

    def get_next_samples(self, n_samples):
        samples, self._t_offset = self.get_samples(n_samples, self._t_offset)
