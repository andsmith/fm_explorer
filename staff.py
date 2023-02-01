"""
Layout of items on staff that don't move.
"""
from clef_drawing import make_bass, make_treble
from music_layout import LAYOUT
import numpy as np
import cv2


class Shape(object):
    PRECISION_BITS = 6

    def __init__(self):
        self._parts = []

    def add_polyline(self, coords, color, thickness, closed, filled):
        coords = (coords * (2. ** Shape.PRECISION_BITS)).astype(np.int32)
        self._parts.append((coords, color, thickness, closed, filled))
        return self

    def add_box(self, x_span, y_span, color, thickness):
        box_points = np.array([[x_span[0], y_span[0]],
                               [x_span[1], y_span[0]],
                               [x_span[1], y_span[1]],
                               [x_span[0], y_span[1]]])
        return self.add_polyline(box_points, color, thickness=thickness, closed=True, filled=False)

    def draw(self, img):
        for coords, color, thickness, closed, filled in self._parts:
            if filled:
                cv2.fillPoly(img, [coords], color, lineType=cv2.LINE_AA, shift=Shape.PRECISION_BITS)
            else:
                cv2.polylines(img, [coords], closed, color, thickness=thickness, lineType=cv2.LINE_AA,
                              shift=Shape.PRECISION_BITS)


class Staff(object):
    def __init__(self, bbox, colors=None):
        self._bbox = bbox
        self._width, self._height = self._bbox['right'] - self._bbox['left'], self._bbox['bottom'] - self._bbox['top']

        # staff area of image
        self._top_y = int(self._bbox['top'] + self._height * LAYOUT['v_span'][0])
        bottom_y = int(self._bbox['top'] + self._height * LAYOUT['v_span'][1])
        self._left_x = int(self._bbox['left'] + self._width * LAYOUT['h_span'][0])
        self._right_x = int(self._bbox['left'] + self._width * LAYOUT['h_span'][1])

        self._width = self._right_x - self._left_x
        height = bottom_y - self._top_y
        self._space = int(height / 10)  # space between ledger lines
        # re-define to avoid aliasing of staff lines
        self._height = self._space * 10
        self._bottom_y = self._top_y + self._height

        self._middle_c_y = ((self._top_y + self._top_y + bottom_y) / 2)

        # thickness of lines
        min_thickness = self._space / 10
        self._line_thickness = int(min_thickness)
        self._note_stem_thickness = int(min_thickness * 1.5)

        # Clef coordinates
        treble_y = self._middle_c_y - self._space * 2.0  # G
        bass_y = self._middle_c_y + self._space * 2.0  # F
        clef_x = self._left_x + LAYOUT['clef_indent_spaces'] * self._space
        treble_pos = np.array((clef_x, treble_y)).reshape(1, -1)
        bass_pos = np.array((clef_x, bass_y)).reshape(1, -1)
        self._bass_clef_coords = [(coords * self._space + bass_pos, is_closed)
                                  for coords, is_closed in zip(*make_bass())]
        self._treble_clef_coords = [(coords * self._space + treble_pos, is_closed)
                                    for coords, is_closed in zip(*make_treble())]

        #  staff lines, vertical, single & double thickness lines on left
        self._light_v_line = np.array([[self._left_x + self._space * 4, self._top_y],
                                       [self._left_x + self._space * 4, self._bottom_y]])
        self._heavy_v_line = np.array([[self._left_x, self._top_y],
                                       [self._left_x, self._bottom_y]])
        # horizontal
        h_line_y = np.linspace(self._top_y, bottom_y, 11).astype(np.int64).tolist()
        line_ys = np.array(h_line_y[:5] + h_line_y[6:])
        self._h_lines = [np.array([[self._left_x, y],
                                   [self._right_x, y]]) for y in line_ys]
        # font = cv2.FONT_HERSHEY_DUPLEX
        # font_scale = 1 + int(spacing / 40)


class Note(object):

    def __init__(self, bbox, colors):

        self._bbox = bbox
        self._width, self._height = self._bbox['right'] - self._bbox['left'], self._bbox['bottom'] - self._bbox['top']
