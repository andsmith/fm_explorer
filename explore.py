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
from sound_tools import sound

from fm_synth import FMSynthesizer
from drawing import AnimatedWave
from instructions import HELP_TEXT


class FMExplorerAppStates(IntEnum):
    idle = 0
    playing = 1


MAX_VOL = 0.8


class FMExplorerApp(object):
    BKG_COLOR = (20, 20, 20, 255)
    MODULATION_COLOR = (64, 64, 255, 255)
    CARRIER_COLOR = (255, 64, 64, 255)
    WAVE_COLOR = (64, 255, 64, 255)
    SPECTRUM_COLOR = (64, 176, 255, 255)
    HELP_COLOR = (240, 240, 240, 255)
    HELP_BKG = (42, 42, 42, 255)
    HELP_FONT = cv2.FONT_HERSHEY_SIMPLEX
    SAMPLING_RATE = 44100
    HELP_OPACITY = None  # 0.9  # None for less CPU
    TRANSLUCENT = 0.5
    TITLE_OPACITY = 0.15

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

    H_DIV_LINE = 0.7  # control and display
    V_DIV_LINE = 0.7  # wave and spectrum

    def __init__(self, window_size,
                 init_carrier_max=(1000.0, MAX_VOL),
                 init_modulation_max=(50.0, 100.),
                 carrier_init=(210., 0.5),
                 modulation_init=(10, 20.)):
        """
        initialize app window
        :param window_size:  Width x height
        """
        self._app_t_start = time.perf_counter()
        self._win_size = window_size
        self._win_name = "FM Explorer"
        self._adjusting_modulation = False
        self._help_display = StatusMessages(window_size[::-1],
                                            text_color=FMExplorerApp.HELP_COLOR,
                                            bkg_color=FMExplorerApp.HELP_BKG, fullscreen=True,
                                            font=FMExplorerApp.HELP_FONT, bkg_alpha=FMExplorerApp.HELP_OPACITY)

        self._help_display.add_msgs(HELP_TEXT, 'help', 0.)

        # init sound & synth
        self._showing_help = True
        self._fm = FMSynthesizer(carrier_init=carrier_init, modulation_init=modulation_init)

        # init gui
        self._n_waveform_samples = int(FMExplorerApp.SAMPLING_RATE / 10.)
        self._blank = np.zeros((window_size[1], window_size[0], 4), np.uint8)
        self._blank[:, :, 3] = 255
        h_div_line = int(self._win_size[1] * FMExplorerApp.H_DIV_LINE)
        v_div_line = int(self._win_size[0] * FMExplorerApp.V_DIV_LINE)
        self._control_bbox = {'top': 0, 'bottom': h_div_line,
                              'left': 0, 'right': window_size[0]}
        self._wave_bbox = {'top': h_div_line, 'bottom': window_size[1],
                           'left': 0, 'right': v_div_line}
        self._spectrum_bbox = {'top': h_div_line, 'bottom': window_size[1],
                               'left': v_div_line, 'right': window_size[0]}
        self._m_grid = CartesianGrid(self._control_bbox, init_values=modulation_init, axis_labels=('F', 'D'),
                                     draw_props={'cursor_string': "(%.2f Hz, %.2f)"},
                                     param_ranges=[[0, init_modulation_max[0]], [0, init_modulation_max[1]]],
                                     colors=FMExplorerApp.M_GRID_COLORS,
                                     title='modulation')
        self._c_grid = CartesianGrid(self._control_bbox, init_values=carrier_init, axis_labels=('F', 'A'),
                                     draw_props={'cursor_string': "(%.2f Hz, %.2f)"},
                                     param_ranges=[[0, init_carrier_max[0]], [0, init_carrier_max[1]]],
                                     colors=FMExplorerApp.C_GRID_COLORS,
                                     title='carrier', adjustability=(True, False))
        self._s_grid = CartesianGrid(self._spectrum_bbox, init_values=(None, None), axis_labels=('F', ' '),
                                     draw_props={'cursor_string': "(%.2f Hz, power = %.2f)",
                                                 'title_font_scale': 1., 'cursor_font_scale': .4,
                                                 'title_thickness': 1},
                                     param_ranges=[[0., 1.], [0., 1.]],
                                     colors=FMExplorerApp.S_GRID_COLORS,
                                     title='power spectrum', adjustability=(False, False))
        self._w_grid = CartesianGrid(self._wave_bbox, init_values=(None, None), axis_labels=('F', None),
                                     draw_props={'cursor_string': None, 'title_font_scale': 1.5,
                                                 'title_thickness': 1,'show_ticks':(True, False)},
                                     param_ranges=[[0., self._n_waveform_samples/FMExplorerApp.SAMPLING_RATE]
                                         , [-.2, 1.2]],
                                     colors=FMExplorerApp.W_GRID_COLORS,
                                     title='waveform', adjustability=(False, False))

        self._wave = AnimatedWave(self._wave_bbox, wave_func=self._make_wave_samples,
                                  bbox_thickness=2,
                                  wave_color=FMExplorerApp.WAVE_COLOR)
        self._mouse_pos = 0, 0
        self._run()

    def _make_wave_samples(self):

        return self._fm.get_samples(self._n_waveform_samples)[0]

    def _mouse(self, event, x, y, flags, param):
        self._mouse_pos = x, y
        # update grid
        if self._adjusting_modulation:
            self._m_grid.mouse(event, x, y, flags, param)
        else:
            self._c_grid.mouse(event, x, y, flags, param)

        self._s_grid.mouse(event, x, y, flags, param)
        self._w_grid.mouse(event, x, y, flags, param)

        # update synth:
        if event == cv2.EVENT_MOUSEMOVE:
            c_freq, c_amp = self._c_grid.get_values()
            m_freq, m_depth = self._m_grid.get_values()
            self._fm.set_params(c_freq=c_freq, c_amp=c_amp, m_freq=m_freq, m_depth=m_depth)

        # handle clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            self._start_playing()
        elif event == cv2.EVENT_LBUTTONUP:
            self._stop_playing()

    def _start_playing(self):
        logging.info("Starting audio.")

    def _stop_playing(self):
        logging.info("Stopping audio.")

    def _make_frame(self):
        frame = self._blank.copy()

        if self._adjusting_modulation:
            self._m_grid.draw(frame)
        else:
            self._c_grid.draw(frame)

        self._s_grid.draw(frame)
        self._w_grid.draw(frame)

        self._wave.draw(frame)
        if self._showing_help:
            # import ipdb; ipdb.set_trace()
            n_chan = 3 if FMExplorerApp.HELP_OPACITY is None else 4
            self._help_display.annotate_img(frame[:, :, :n_chan])

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

    def _keyboard(self,k):

        # send keystroke to appropriate grid
        if in_bbox(self._control_bbox, self._mouse_pos):

            if self._adjusting_modulation:
                _ = self._m_grid.keyboard(k)
                m_freq, m_depth = self._m_grid.get_values()
                self._fm.set_params(c_freq=None, c_amp=None, m_freq=m_freq, m_depth=m_depth)
            else:  # adjusting carrier
                _ = self._c_grid.keyboard(k)
                c_freq, c_amp = self._c_grid.get_values()
                self._fm.set_params(c_freq=c_freq, c_amp=c_amp, m_freq=None, m_depth=None)

        elif in_bbox(self._wave_bbox, self._mouse_pos):
            _ = self._w_grid.keyboard(k)

        elif in_bbox(self._spectrum_bbox, self._mouse_pos):
            new_spectrum_ranges = self._s_grid.keyboard(k)

        if k & 0xff == ord('q'):
            return True
        if k & 0xff == ord('h'):
            self._showing_help = not self._showing_help
        elif k & 0xff == ord(' '):
            self._adjusting_modulation = not self._adjusting_modulation
        elif k & 0xff == ord(','):
            self._n_waveform_samples = int(self._n_waveform_samples * 0.75)
            self._w_grid.set_param_range(self._n_waveform_samples/FMExplorerApp.SAMPLING_RATE, 0)
        elif k & 0xff == ord('.'):
            self._n_waveform_samples = int(self._n_waveform_samples * 1.25)
            self._w_grid.set_param_range(self._n_waveform_samples/FMExplorerApp.SAMPLING_RATE, 0)
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    FMExplorerApp((1000, 800))
