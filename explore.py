"""
Main app

See http://www.cs.cmu.edu/~music/icm-online/readings/fm-synthesis/index.html

Use the test args from that page to re-generate its plot:

                 carrier_init=(210., 0.5)
                 modulation_init=(35., 10.)
"""
from enum import IntEnum
import cv2
import numpy as np
import logging
import time

from gui_utils.text_annotation import StatusMessages
from gui_utils.coordinate_grids import CartesianGrid
from gui_utils.drawing import blend_colors, in_bbox
from sound_tools.sound_player import SoundPlayer, Encoder

from sound_tools.fm_synth import FMSynthesizer
from drawing import AnimatedWave, AnimatedSpectrum
from instructions import HELP_TEXT
from staff import Staff


class FMExplorerAppStates(IntEnum):
    adjusting_carrier = 0
    playing_theremin = 1


MAX_VOL = 0.8
QUIET_AMP = 0.0001


class FMExplorerApp(object):
    BKG_COLOR = (20, 20, 20, 255)
    MODULATION_COLOR = (64, 64, 255, 255)
    CARRIER_COLOR = (255, 64, 64, 255)
    WAVE_COLOR = (64, 255, 64, 255)
    SPECTRUM_COLOR = (64, 176, 255, 255)
    HELP_COLOR = (240, 240, 240, 255)
    HELP_BKG = (42, 42, 42, 240)
    STATUS_MSG_BOX_SIZE = 300, 100
    HELP_FONT = cv2.FONT_HERSHEY_SIMPLEX
    SAMPLING_RATE = 44100
    HELP_OPACITY = 0.97  # = None for less CPU
    TRANSLUCENT = 0.5
    TITLE_OPACITY = 0.25

    WINDOW_SEPARATION = 15
    M_GRID_COLORS = {'heavy': MODULATION_COLOR,
                     'light': blend_colors(MODULATION_COLOR, BKG_COLOR, TRANSLUCENT).tolist(),
                     'title': blend_colors(MODULATION_COLOR, BKG_COLOR, TITLE_OPACITY).tolist(),
                     'bkg': BKG_COLOR}

    C_GRID_COLORS = {'heavy': CARRIER_COLOR,
                     'light': blend_colors(CARRIER_COLOR, BKG_COLOR, TRANSLUCENT).tolist(),
                     'title': blend_colors(CARRIER_COLOR, BKG_COLOR, TITLE_OPACITY).tolist(),
                     'bkg': BKG_COLOR}

    S_GRID_COLORS = {'heavy': SPECTRUM_COLOR,
                     'light': blend_colors(SPECTRUM_COLOR, BKG_COLOR, TRANSLUCENT).tolist(),
                     'title': blend_colors(SPECTRUM_COLOR, BKG_COLOR, TITLE_OPACITY).tolist(),
                     'bkg': BKG_COLOR}

    W_GRID_COLORS = {'heavy': WAVE_COLOR,
                     'light': blend_colors(WAVE_COLOR, BKG_COLOR, TRANSLUCENT).tolist(),
                     'title': blend_colors(WAVE_COLOR, BKG_COLOR, TITLE_OPACITY).tolist(),
                     'bkg': BKG_COLOR}
    CONTROL_DIV_LINE = 0.5
    H_DIV_LINE = 0.7  # control and display
    V_DIV_LINE = 0.4  # wave and spectrum

    AUDIO_PARAMS = dict(channels=1, sample_width=2, frame_rate=44100)

    def __init__(self, window_size,
                 init_carrier_max=(1000.0, MAX_VOL),
                 init_modulation_max=(400.0, 25.),
                 carrier_init=(440., 0.31),
                 modulation_init=(100, 0.)):
        """
        initialize app window
        :param window_size:  Width x height
        """
        # state
        self._playing = False
        self._state = FMExplorerAppStates.adjusting_carrier
        self._showing_help = False
        self._timbre_mode = False  # change modulation with carrier to preserve timbre
        self._show_box = False  # box around note-area of staff
        self._autotune = False
        self._last_window = None

        # set self._volume= carrier_amp when switching to theremin mode (where vol = self._volume * car_freq)
        self._volume = None

        self._app_t_start = time.perf_counter()
        self._win_size = window_size
        self._win_name = "FM Explorer"

        # init gui layout & colors
        self._n_waveform_samples = int(FMExplorerApp.SAMPLING_RATE / 15.)
        self._blank = np.zeros((window_size[1], window_size[0], 4), np.uint8)
        self._blank[:, :, 3] = 255

        # define different bounding boxes
        h_div_line = int(self._win_size[1] * FMExplorerApp.H_DIV_LINE)
        ctrl_div_line = int(self._win_size[0] * FMExplorerApp.CONTROL_DIV_LINE)
        v_div_line = int(self._win_size[0] * FMExplorerApp.V_DIV_LINE)

        spectrum_f_range, spectrum_p_range = [0.0, 6000.0], (0., 1.)
        s = FMExplorerApp.WINDOW_SEPARATION

        self._modulation_box = {'top': s * 2, 'bottom': h_div_line - s,
                                'left': s * 2, 'right': ctrl_div_line - s}

        self._carrier_box = {'top': s * 2, 'bottom': h_div_line - s,
                             'left': ctrl_div_line + s, 'right': window_size[0] - s * 2}

        self._wave_bbox = {'top': h_div_line + s, 'bottom': window_size[1] - s * 2,
                           'left': s * 2, 'right': v_div_line - s}

        self._spectrum_bbox = {'top': h_div_line + s, 'bottom': window_size[1] - s * 2,
                               'left': v_div_line + s, 'right': window_size[0] - s * 2}

        self._m_grid = CartesianGrid(self._modulation_box, init_values=modulation_init, axis_labels=('F', 'D'),
                                     draw_props={'cursor_string': "(%.2f Hz, %.2f)"},
                                     param_ranges=[[0, init_modulation_max[0]], [0, init_modulation_max[1]]],
                                     colors=FMExplorerApp.M_GRID_COLORS,
                                     title='modulation')
        self._c_grid = CartesianGrid(self._carrier_box, init_values=carrier_init, axis_labels=('F', 'A'),
                                     draw_props={'cursor_string': "(%.2f Hz, %.2f)"},
                                     param_ranges=[[0, init_carrier_max[0]], [0, init_carrier_max[1]]],
                                     colors=FMExplorerApp.C_GRID_COLORS,
                                     title='carrier', adjustability=(True, False))

        self._s_grid = CartesianGrid(self._spectrum_bbox, init_values=(None, None), axis_labels=('F (Hz)', 'log(p)'),
                                     draw_props={'cursor_string': None,
                                                 'title_font_scale': 1.4, 'cursor_font_scale': .4,
                                                 'axis_font_scale': .4,
                                                 'title_thickness': 1, 'show_ticks': (True, False),
                                                 'user_marker': False},
                                     param_ranges=[spectrum_f_range, spectrum_p_range],
                                     colors=FMExplorerApp.S_GRID_COLORS,
                                     title='power spectrum', adjustability=(True, False))

        self._w_grid = CartesianGrid(self._wave_bbox, init_values=(None, None), axis_labels=('T (sec.)', None),
                                     draw_props={'cursor_string': None, 'title_font_scale': 1.5,
                                                 'title_thickness': 1, 'show_ticks': (True, False),
                                                 'user_marker': False},
                                     param_ranges=[[0., self._n_waveform_samples / FMExplorerApp.SAMPLING_RATE]
                                         , [-.2, 1.2]],
                                     colors=FMExplorerApp.W_GRID_COLORS,
                                     title='waveform', adjustability=(False, False))
        # give C and M grids initial focus
        self._mouse_pos = int((self._modulation_box['left'] + self._modulation_box['right']) / 2), \
                          int((self._modulation_box['top'] + self._modulation_box['bottom']) / 2)
        self._c_grid.mouse(cv2.EVENT_MOUSEMOVE, self._mouse_pos[0], self._mouse_pos[1], None, None)
        self._m_grid.mouse(cv2.EVENT_MOUSEMOVE, self._mouse_pos[0], self._mouse_pos[1], None, None)

        # init sound & synth & music
        self._encoder = Encoder(FMExplorerApp.AUDIO_PARAMS['sample_width'])
        self._staff = Staff(self._carrier_box)

        # things with a .mouse() method with the cv2 callback params
        self._windows = {'carrier': self._c_grid,
                         'mod': self._m_grid,
                         'wave': self._w_grid,
                         'spectrum': self._s_grid,
                         'staff': self._staff}
        self._fm = FMSynthesizer(rate=FMExplorerApp.SAMPLING_RATE)
        self._update_synth()
        self._audio = SoundPlayer(sample_generator=lambda n: self._fm.get_samples(n, encode_func=self._encoder.encode),
                                  **FMExplorerApp.AUDIO_PARAMS)

        # help
        msg_params = dict(
            text_color=FMExplorerApp.HELP_COLOR, v_anchor='both',
            bkg_color=FMExplorerApp.HELP_BKG, inside_margins=(20, 20), outside_margins=(20, 20),
            font=FMExplorerApp.HELP_FONT)
        self._help_display = StatusMessages(window_size[::-1], **msg_params)
        self._help_display.add_msgs(HELP_TEXT, 'help', 0.)

        # status msgs
        msg_params['outside_margins'] = (
                (np.array(window_size) - np.array(FMExplorerApp.STATUS_MSG_BOX_SIZE)) / 2).astype(np.int64)
        self._status_display = StatusMessages(window_size[::-1], **msg_params)
        # init animations
        self._spectrum = AnimatedSpectrum(self._spectrum_bbox, f_range=spectrum_f_range,
                                          color=FMExplorerApp.SPECTRUM_COLOR)
        self._wave = AnimatedWave(self._wave_bbox, color=FMExplorerApp.WAVE_COLOR)
        self._set_animation_samples()

        self._run()

    def _set_animation_samples(self):

        samples = self._fm.get_plot_samples(self._n_waveform_samples)

        self._wave.set_samples(samples)
        self._spectrum.set_samples(samples)

    def _mouse(self, event, x, y, flags, param):
        self._mouse_pos = x, y

        if self._showing_help:
            return

        # send signals to correct window, the last one the mouse was in
        if in_bbox(self._modulation_box, self._mouse_pos):
            self._last_window = 'mod'
        elif in_bbox(self._wave_bbox, self._mouse_pos):
            self._last_window = 'wave'
        elif in_bbox(self._spectrum_bbox, self._mouse_pos):
            self._last_window = 'spectrum'
        elif in_bbox(self._carrier_box, self._mouse_pos):
            if self._state == FMExplorerAppStates.adjusting_carrier:
                self._last_window = 'carrier'
            elif self._state == FMExplorerAppStates.playing_theremin:
                self._last_window = 'staff'
        if self._last_window is None:
            # no one to tell
            return

        if event==cv2.EVENT_LBUTTONUP:
            # send to all windows, should be safe to unpress if they're not traccking a press anyway
            for w in self._windows:
                self._windows[w].mouse(event,x, y, flags, param)

        # remember to see if changes are made
        old_carrier_freq, old_carrier_amp = self._c_grid.get_values()

        # send signal to the correct window
        note_info = self._windows[self._last_window].mouse(event, x, y, flags, param)

        # if values changed on the staff, update the grid(s) too
        if self._state == FMExplorerAppStates.playing_theremin:
            if self._last_window == 'staff':

                # set theremin freq
                if self._autotune:
                    freq = note_info['autotune_frequency']
                else:
                    freq = note_info['frequency']

                # set theremin amplitude depending on last carrier amp, current button state, and current staff amp
                if note_info['pushed']:
                    amp = note_info['amplitude'] * self._volume
                else:
                    amp = QUIET_AMP

                self._c_grid.move_marker((freq, amp))

        # adjust for timbre
        new_carrier_freq, new_carrier_amp = self._c_grid.get_values()

        if self._timbre_mode and old_carrier_freq != new_carrier_freq:
            old_mod_freq, old_mod_amp = self._m_grid.get_values()
            new_modulation_freq = old_mod_freq / old_carrier_freq * new_carrier_freq if old_carrier_freq > 0 else 0
            self._m_grid.move_marker((new_modulation_freq, old_mod_amp))

        # update synth
        self._update_synth()
        self._set_animation_samples()

    def _update_synth(self):
        c = self._c_grid.get_values()
        m = self._m_grid.get_values()

        params = {'carrier_freq': c[0], 'carrier_amp': c[1],
                  'mod_freq': m[0], 'mod_depth': m[1]}
        self._fm.set_params(params)

    def _make_frame(self):
        frame = self._blank.copy()

        # add grids to frame
        self._m_grid.draw(frame)
        self._s_grid.draw(frame)
        self._w_grid.draw(frame)
        if self._state == FMExplorerAppStates.adjusting_carrier:
            self._c_grid.draw(frame)
        else:
            self._staff.draw(frame, show_box=self._show_box)

        # add animations
        self._wave.draw(frame)
        self._spectrum.draw(frame)

        if self._showing_help:
            n_chan = 3 if FMExplorerApp.HELP_OPACITY is None else 4
            self._help_display.annotate_img(frame[:, :, :n_chan])

        # on top
        self._status_display.annotate_img(frame)

        return frame

    def _run(self):
        # init display & mouse
        cv2.namedWindow(self._win_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self._win_name, self._mouse)

        t_start = time.perf_counter()
        n_frames = 0
        while True:
            frame = self._make_frame()
            cv2.imshow(self._win_name, frame)
            k = cv2.waitKey(1)
            if self._keyboard(k):
                break
            n_frames += 1
            duration = time.perf_counter() - t_start
            if duration > 5.0:
                fps = n_frames / duration
                logging.info("FPS:  %.3f" % (fps,))
                n_frames = 0
                t_start = time.perf_counter()
        if self._playing:
            self._audio.stop()

    def _keyboard(self, k):

        # send keystroke to appropriate grid
        if in_bbox(self._modulation_box, self._mouse_pos):
            _ = self._m_grid.keyboard(k)
        elif self._state == FMExplorerAppStates.adjusting_carrier and in_bbox(self._carrier_box, self._mouse_pos):
            _ = self._c_grid.keyboard(k)
        elif in_bbox(self._spectrum_bbox, self._mouse_pos):
            new_param_range = self._s_grid.keyboard(k)  # might change here, update animation
            if new_param_range is not None:
                self._spectrum.set_f_range(new_param_range[0, :], new_param_range[1, :])

        # change states
        if k & 0xff == ord('c'):
            self._state = FMExplorerAppStates.adjusting_carrier
            old_carrier_freq, _ = self._c_grid.get_values()
            self._c_grid.move_marker((old_carrier_freq, QUIET_AMP))
            self._last_window = None
        if k & 0xff == ord('t'):
            self._state = FMExplorerAppStates.playing_theremin
            old_carrier_freq, self._volume = self._c_grid.get_values()
            self._c_grid.move_marker((old_carrier_freq, QUIET_AMP))
            self._update_synth()
            self._last_window = None
            # send fake mouse event to self
        # general keystrokes
        if k & 0xff == ord('q'):
            return True

        elif k & 0xff == ord('x'):
            self._timbre_mode = not self._timbre_mode
            t_str = "Timbre mode:  %s" % ('ON' if self._timbre_mode else 'OFF',)
            logging.info(t_str)
            self._status_display.add_msg(t_str, 'timbre_mode', duration_sec=1.0)

        elif k & 0xff == ord('h'):
            self._showing_help = not self._showing_help

        elif k & 0xff == ord('b'):
            self._show_box = not self._show_box

        elif k & 0xff == ord('a'):
            self._autotune = not self._autotune
            t_str = "Autotune:  %s" % ('ON' if self._autotune else 'OFF',)
            logging.info(t_str)
            self._status_display.add_msg(t_str, 'autotune', duration_sec=1.0)


        elif k & 0xff == ord(' '):
            if not self._playing:
                logging.info("Starting sound")
                self._playing = True
                self._audio.start()
            else:
                logging.info("Stopping sound")
                self._audio.stop()
                self._fm.reset_state()
                self._playing = False

        # changing number of samples in wave, update grids and animations
        elif k & 0xff == ord(','):
            self._n_waveform_samples = int(self._n_waveform_samples * 0.75)
            self._w_grid.set_param_max(self._n_waveform_samples / FMExplorerApp.SAMPLING_RATE, 0)

            self._set_animation_samples()
        elif k & 0xff == ord('.'):
            self._n_waveform_samples = int(self._n_waveform_samples * 1.25)
            self._w_grid.set_param_max(self._n_waveform_samples / FMExplorerApp.SAMPLING_RATE, 0)
            self._set_animation_samples()

        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    FMExplorerApp((900, 700))
