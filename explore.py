"""
Main app
"""
from enum import IntEnum
import cv2
import numpy as np
import logging
import time

from gui_utils.coordinate_grids import CartesianGrid


class FMExplorerAppStates(IntEnum):
    idle = 0
    playing = 1


class FMExplorerApp(object):

    def __init__(self, window_size, init_modulation_range=(1000.0, 100.)):
        """
        initialize app window
        :param window_size:  Width x height
        :param init_modulation_range:  (max modulation freq, max amp.)
        """
        self._win_size = window_size
        self._win_name = "FM Explorer"

        self._blank = np.zeros((window_size[1], window_size[0], 4), np.uint8)
        self._blank[:, :, 3] = 255

        self._grid = CartesianGrid({'top': 0, 'bottom': window_size[1],
                                    'left': 0, 'right': window_size[0]},
                                   param_ranges=[[0.,init_modulation_range[0]],
                                                 [0.,init_modulation_range[1]]])
        self._run()

    def _mouse(self, event, x, y, flags, param):
        # update grid
        self._grid.mouse(event, x, y, flags, param)

        # handle clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            self._start_playing()
        elif event == cv2.EVENT_LBUTTONUP:
            self._stop_playing()

    def _start_playing(self):
        logging.info("Starting audio.")

    def _stop_playing(self):
        logging.info("Stopping audio.")

    def _run(self):

        cv2.namedWindow(self._win_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self._win_name, self._mouse)
        t_start = time.perf_counter()
        n_frames =0
        while True:
            frame = self._blank.copy()

            self._grid.draw(frame)
            self._grid.draw_crosshair(frame)

            cv2.imshow(self._win_name, frame)
            k = cv2.waitKey(1)
            self._grid.keyboard(k)
            if k & 0xff == ord('q'):
                break

            n_frames+=1
            duration = time.perf_counter() - t_start
            if duration > 5.0:
                fps = n_frames/duration
                logging.info("FPS:  %.3f" % (fps, ))
                n_frames=0
                t_start = time.perf_counter()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    FMExplorerApp((1000, 800))
