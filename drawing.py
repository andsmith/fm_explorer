import numpy as np
import time
import cv2
from gui_utils.drawing import draw_rect,draw_box

class AnimatedWave(object):
    def __init__(self, bbox, wave_func, bbox_thickness=2, wave_color=(255, 255, 255, 255)):
        self._bbox = bbox
        self._wave_func = wave_func
        self._wave_color = wave_color
        self._last_params = None
        self._last_coords = None
        self._precision_bits = 6
        self._bbox_thickness = bbox_thickness
        self._precision_mult = 2 ** self._precision_bits
        self._x_values = None
        self._t_start =time.perf_counter()

    def draw(self, image, params=None):
        """
        Render wave in image.  Params optional, used to avoid re-calculating coords.
        """
        #if time.perf_counter()-self._t_start > 2.0:
        #    import ipdb;
        #    ipdb.set_trace()
        # Draw bounding box:
        if self._bbox_thickness> 0:
            draw_box(image, self._bbox,self._bbox_thickness, self._wave_color)

        # update coords
        if self._last_params is None or params is None or self._last_params != params:
            self._last_coords = self._get_wave_image_coords()
        self._last_params = params

        # draw
        cv2.polylines(image, [self._last_coords], False, self._wave_color, 1, cv2.LINE_AA, self._precision_bits)

    def _get_wave_image_coords(self):
        """
        How to render a wave in a given bounding box of an image
        """

        samples = self._wave_func()
        y_range = self._bbox['bottom'] - self._bbox['top']

        y_values = (-samples / 2. + 0.5) * y_range + self._bbox['top']
        if self._x_values is None or self._x_values.size!=samples.size:
            self._x_values = np.linspace(self._bbox['left'], self._bbox['right'] - 1, samples.size)

        return (self._precision_mult * np.dstack([self._x_values, y_values])).reshape(-1, 1, 2).astype(np.int32)


def test_wave():
    slowdown = 1000.0
    frequency = 220.0

    t_start = time.perf_counter()

    def _get_t_offset():
        dt = (time.perf_counter() - t_start)
        if int(dt) % 4 <= 1:
            dt = 0
        else:
            dt /= slowdown
        return dt

    def _wave_func(n):
        t = np.linspace(0.0, n / 44100., n + 1)[:-1] + _get_t_offset()

        return np.sin(np.pi * 2. * frequency * t)

    bkg = np.zeros((500, 750, 4), dtype=np.uint8)
    bkg[:, :, 3] = 255
    bbox = {'top': 0, 'left': 0, 'right': bkg.shape[1], 'bottom': bkg.shape[0]}
    wave = AnimatedWave(bbox, _wave_func)

    fps_t_start = time.perf_counter()
    n_frames = 0
    draw_times = []
    while True:
        frame = bkg.copy()
        draw_start = time.perf_counter()
        wave.draw(frame, params=_get_t_offset())
        draw_times.append(time.perf_counter() - draw_start)
        cv2.imshow("Wave - q to quit", frame)
        k = cv2.waitKey(1)
        if k & 0xff == ord('q'):
            break

        n_frames += 1
        elapsed = time.perf_counter() - fps_t_start
        if elapsed > .5:
            draw_fps = 1.0 / np.array(draw_times)
            print("Mean FPS:  %.1f   (sd.  %.3f)" % (float(np.mean(draw_fps)), float(np.std(draw_fps))))
            fps_t_start += elapsed
            draw_times = []


if __name__ == "__main__":
    test_wave()
