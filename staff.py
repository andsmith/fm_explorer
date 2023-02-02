"""
Basic musical typesetting.
"""
import logging
from drawing import eval_ellipse
from clef_drawing import make_bass, make_treble

import numpy as np
import cv2
import time

PRECISION_BITS = 6  # drawing
DEFAULT_COLORS_BGRA = {'bkg': (229, 235, 245, 255),
                       'lines': (3, 7, 9, 255),
                       'text': (3, 7, 9, 255),
                       'notes': (3, 7, 9, 255),
                       'mouse': (103, 107, 109, 255),
                       'guide_box': (64, 255, 64, 255),
                       'volume': (32, 200, 32, 255)}
NOTE_ECCENTRICITY = 0.75
NOTE_ROTATION_DEG = 15.0
LAYOUT = {'staff_v_span': [.25, .7],  # all dims relative to bbox  (unit)
          'staff_h_span': [.1, .9],
          'clef_indent_spaces': 1.75,
          'wedge_length': 0.33,
          'title': {'txt': 'Theremin',
                    'font': cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    'font_scale_mult': 1 / 20},  # times note space
          'dynamics': {'font': cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                       'font_scale_mult': 1 / 35},
          'ledger_line_length_mult': 1.5,
          'n_max_ledger_lines': (5, 5)}  # above and below


class NoteArea(object):
    """
    Draw a note as the mouse cursor
    """

    def __init__(self, bbox, middle_c_y, space, staff_line_thickness, colors=None):
        """
        Define note drawing params.
        :param bbox:  bounding box of note area in image, dict with 'top', 'bottom','left','right' and int values
        :param middle_c_y:  y-value of middle-c (float)
        :param space:  y-distance between ledger lines (i.e. definition of whole step)
        :param staff_line_thickness:  note size depends on this to fit exactly between lines (float)
        """
        self._colors = colors if colors is not None else DEFAULT_COLORS_BGRA
        self.c_y = middle_c_y
        self.space = space

        self._autotuning = False

        self._mouse_pos_xy = None
        self._note_pos_xy = None
        self.pushed = False
        self._bbox = bbox
        self._staff_line_thickness = staff_line_thickness
        self._note_stem_thickness = int(staff_line_thickness * 1.5)

        self._set_geom()

        # derived from position
        self.steps_to_middle_c = None
        self.show_middle_ledger_line = False
        self.amplitude = 0.0
        self.frequency = 0.0

    def toggle_autotune(self):
        self._autotuning = not self._autotuning
        logging.info("Autotune:  %s" % (self._autotuning,))

    def push(self):

        self.pushed = True

    def unpush(self):

        self.pushed = False

    def set_pos(self, x, y):
        """
        Set the position of the note on the staff.
        Determine frequency & amplitude.
        NOTE:  Clip to bbox
        """
        x = np.clip(x, self._bbox['left'], self._bbox['right'])
        y = np.clip(y, self._bbox['top'], self._bbox['bottom'])
        self._mouse_pos_xy = np.array((x, y))

        # set position on staff, number of ledger lines
        self.steps_to_middle_c = (y - self.c_y) / self.space
        self.n_ledger_lines = int(np.ceil(np.abs(self.steps_to_middle_c) - 5.5))
        if np.abs(self.steps_to_middle_c) < 5.5:
            self.n_ledger_lines = 0
        self.show_middle_ledger_line = np.abs(self.steps_to_middle_c) < 0.49
        width = self._bbox['right'] - self._bbox['left']

        # notestem info

        # if np.abs(self.steps_to_middle_c) > 6.5:
        # assert self.n_ledger_lines >= 2, "Bad ledger line calc, should have at least two!"
        # self._stem_length = -(self.steps_to_middle_c) * self.space
        # else:
        self._stem_length = 3.5 * self.space * -np.sign(self.steps_to_middle_c)

        # get audio info
        self.amplitude = np.clip((x - self._bbox['left']) / width, 0, 1.0)
        self.frequency = 2. ** (self.steps_to_middle_c * 2. / 12.)

        self._note_pos_xy = self._mouse_pos_xy.copy()
        self._autotune()


    def _autotune(self):
        # set note y position backwards from frequency if changing it
        pass

    def _set_geom(self, n_pts=50):
        width, height = self._bbox['right'] - self._bbox['left'], \
                        self._bbox['bottom'] - self._bbox['top']
        # note shape
        t = np.linspace(0, np.pi * 2, n_pts)  # increase for better note shape
        phi_rad = np.deg2rad(NOTE_ROTATION_DEG)
        h = self.space / 2. - self._staff_line_thickness / 2.  # note half-height
        self._coords = eval_ellipse(t, NOTE_ECCENTRICITY, phi_rad, h)  # note, slightly off-center
        self._note_width = np.max(self._coords[:, 0]) - np.min(self._coords[:, 0])
        self._ledger_line_width = self._note_width * 1.5

        # stem attachment points, right & left most point of note shape
        left_most = np.min(self._coords[:, 0])
        all_lefties = np.where(self._coords[:, 0] == left_most)[0]
        self._left_attach_pos = np.mean(self._coords[all_lefties, :], axis=0)
        right_most = np.max(self._coords[:, 0])
        all_righties = np.where(self._coords[:, 0] == right_most)[0]
        self._right_attach_pos = np.mean(self._coords[all_righties, :], axis=0)

        # ledger-line locations,
        self._ledger_y_offsets = np.arange(6, 6 + height / self.space) * self.space

    def draw(self, frame):
        if self._note_pos_xy is None:
            return

        # note head and stem
        note_coords = ((self._coords + self._note_pos_xy) * 2 ** PRECISION_BITS).astype(np.int32)
        note_attach_pos = self._left_attach_pos + self._note_stem_thickness / 2 if self.steps_to_middle_c < 0 else self._right_attach_pos - self._note_stem_thickness / 2
        stem_coords = [self._note_pos_xy + note_attach_pos]
        stem_coords.append([stem_coords[0][0], stem_coords[0][1] + self._stem_length])
        stem_coords = (np.array(stem_coords) * 2. ** PRECISION_BITS).astype(np.int32)
        color = self._colors['notes'] if self.pushed else self._colors['mouse']
        cv2.fillPoly(frame, [note_coords], color, lineType=cv2.LINE_AA, shift=PRECISION_BITS)
        cv2.polylines(frame, [stem_coords], True, color, thickness=self._note_stem_thickness, lineType=cv2.LINE_AA,
                      shift=PRECISION_BITS)

        # ledger lines
        def _get_ledger_line_at(x, y):
            return (np.array([[x - self._ledger_line_width / 2, y],
                              [x + self._ledger_line_width / 2, y]]) * 2. ** PRECISION_BITS).astype(np.int32)

        ledger_lines = []
        if self.show_middle_ledger_line:
            ledger_lines.append(_get_ledger_line_at(self._note_pos_xy[0], self.c_y))
        direction = np.sign(self.steps_to_middle_c)
        ledger_lines.extend([_get_ledger_line_at(self._note_pos_xy[0], self._ledger_y_offsets[i] * direction + self.c_y)
                             for i in range(self.n_ledger_lines)])
        cv2.polylines(frame, ledger_lines, True, color, thickness=self._note_stem_thickness,
                      lineType=cv2.LINE_AA, shift=PRECISION_BITS)


class Staff(object):
    """
    Draw treble & bass staves & clefs.
    Draw crescendo wedge under.
    """

    def __init__(self, bbox, colors=None):
        self._bbox = bbox
        self._colors = colors if colors is not None else DEFAULT_COLORS_BGRA
        self._width = self._bbox['right'] - self._bbox['left']
        self._height = self._bbox['bottom'] - self._bbox['top']

        self._calc_staff_dims()

        # save these for fast drawing
        self._normal_line_key = (self._colors['lines'], max(1, int(self._staff_line_thickness - 1)), False)
        self._lines = {
            self._normal_line_key: []}  # key is (color, thickness, closed)-tuple (2 below), value is list of arrays of points to join

        self._rects = []  # list of dicts {'coords':, 'color':}

        self._set_staff_lines()
        self._set_clef_lines()
        self._set_wedge_lines()
        self._set_text_positions()

        self._note_bbox = {'top': np.max((self._bbox['top'],
                                          self._middle_c_y - self._space * (5 + LAYOUT['n_max_ledger_lines'][1]))),
                           'bottom': np.min((self._bbox['bottom'],
                                             self._middle_c_y + self._space * (5 + LAYOUT['n_max_ledger_lines'][1]))),
                           'left': self._wedge_left[0],
                           'right': self._wedge_right[0]}
        # for drawing note area
        self._note_box_coords = [np.array([[self._note_bbox['left'], self._note_bbox['bottom']],
                                           [self._note_bbox['left'], self._note_bbox['top']],
                                           [self._note_bbox['right'], self._note_bbox['top']],
                                           [self._note_bbox['right'], self._note_bbox['bottom']]], dtype=np.int32)]

        self._note = NoteArea(self._note_bbox, self._middle_c_y, self._space,
                              self._staff_line_thickness, colors=self._colors)

        self._mouse_pos = None
        self._button_down = None

        # convert all coordinates
        precision_mult = 2. ** PRECISION_BITS
        for key in self._lines:
            self._lines[key] = [np.int32(coords * precision_mult) for coords in self._lines[key]]

    def _set_text_positions(self):
        self._title_thickness = max(2, int(self._staff_line_thickness / 2))
        self._dynamics_thickness = max(1, self._title_thickness - 1)
        self._title_font_scale = self._space * LAYOUT['title']['font_scale_mult']
        self._dynamics_font_scale = self._space * LAYOUT['dynamics']['font_scale_mult']
        (w, h), _ = cv2.getTextSize(LAYOUT['title']['txt'], LAYOUT['title']['font'], self._title_font_scale,
                                    self._title_thickness)
        x = int((self._width - w) / 2)
        y = int(self._bbox['top'] + self._space + h)
        self._title_pos = x, y

        (ppp_w, ppp_h), _ = cv2.getTextSize('ppp', LAYOUT['dynamics']['font'], self._dynamics_font_scale,
                                            self._dynamics_thickness)
        wedge_to_text_space = max(1, int(self._space / 1.5))
        self._ppp_pos = (self._wedge_left[0] - ppp_w - wedge_to_text_space, self._wedge_right[1])
        self._fff_pos = (self._wedge_right[0] + wedge_to_text_space + 3, self._wedge_right[1])

    def _calc_staff_dims(self):
        staff_bbox_top = int(self._bbox['top'] + self._height * LAYOUT['staff_v_span'][0])
        staff_bbox_bottom = int(self._bbox['top'] + self._height * LAYOUT['staff_v_span'][1])
        staff_bbox_left = int(self._bbox['left'] + self._width * LAYOUT['staff_h_span'][0])
        staff_bbox_right = int(self._bbox['left'] + self._width * LAYOUT['staff_h_span'][1])

        # re-define height to avoid aliasing of staff lines
        self._staff_width = staff_bbox_right - staff_bbox_left
        staff_height = staff_bbox_bottom - staff_bbox_top
        self._space = int(staff_height / 10)  # space between ledger lines
        staff_height = self._space * 10
        staff_bbox_bottom = staff_bbox_top + staff_height
        self._staff_bbox = {'top': staff_bbox_top,
                            'bottom': staff_bbox_bottom,
                            'left': staff_bbox_left,
                            'right': staff_bbox_right}

        self._staff_line_thickness = int(self._space / 8)

        self._middle_c_y = ((self._staff_bbox['top'] + self._staff_bbox['bottom']) / 2)  # should make integer?

    def _set_staff_lines(self):
        #  staff lines, vertical, single & double thickness lines on left
        # are rects to keep from looking sloppy

        # thick vert. line
        self._rects = [{'coords': [[self._staff_bbox['top'],
                                    self._staff_bbox['bottom']],
                                   [self._staff_bbox['left'],
                                    self._staff_bbox['left'] + self._staff_line_thickness * 2]],
                        'color': self._colors['lines']}]
        # thin vert. line
        self._rects.append({'coords': [[self._staff_bbox['top'],
                                        self._staff_bbox['bottom']],
                                       [self._staff_bbox['left'] + self._staff_line_thickness * 3,
                                        self._staff_bbox['left'] + self._staff_line_thickness * 4]],
                            'color': self._colors['lines']})

        # the ten staff lines
        h_line_y = np.linspace(self._staff_bbox['top'], self._staff_bbox['bottom'], 11).astype(np.int64).tolist()
        line_ys = np.array(h_line_y[:5] + h_line_y[6:])
        self._rects.extend([{'coords': [[y, y + self._staff_line_thickness],
                                        [self._staff_bbox['left'], self._staff_bbox['right']]],
                             'color': self._colors['lines']} for y in line_ys])

    def _set_clef_lines(self):
        # Clef coordinates
        treble_y = self._middle_c_y - self._space * 2.0  # G
        bass_y = self._middle_c_y + self._space * 2.0  # F
        clef_x = self._staff_bbox['left'] + LAYOUT['clef_indent_spaces'] * self._space
        treble_pos = np.array((clef_x, treble_y)).reshape(1, -1)
        bass_pos = np.array((clef_x, bass_y)).reshape(1, -1)
        bass = make_bass()
        treble = make_treble()

        self._lines[self._normal_line_key].extend([coords * self._space + bass_pos
                                                   for coords, closed in zip(*bass)])

        self._lines[self._normal_line_key].extend([coords * self._space + treble_pos
                                                   for coords, closed in zip(*treble)])

    def _set_wedge_lines(self):
        # crescendo wedge
        center_x = (self._staff_bbox['left'] + self._staff_bbox['right']) / 2
        wedge_left_x = int(center_x - self._width * LAYOUT['wedge_length'] / 2)
        wedge_right_x = int(center_x + self._width * LAYOUT['wedge_length'] / 2)
        wedge_y_center = self._staff_bbox['bottom'] + int(self._space * 4)
        wedge_right_ys = wedge_y_center - self._space * .5, \
                         wedge_y_center + self._space * .5
        self._wedge_left = wedge_left_x, wedge_y_center
        self._wedge_right = wedge_right_x, wedge_y_center
        wedge_right_up = wedge_right_x, wedge_right_ys[0]
        wedge_right_down = wedge_right_x, wedge_right_ys[1]
        self._lines[self._normal_line_key].append(np.array([wedge_right_down, self._wedge_left, wedge_right_up]))

        self._volume_left = np.array((wedge_left_x + self._staff_line_thickness / 2, wedge_y_center))

        self._volume_right_up = np.array((wedge_right_x, wedge_right_ys[0] + self._staff_line_thickness ))
        self._volume_right_down = np.array((wedge_right_x, wedge_right_ys[1] - self._staff_line_thickness))

    def draw(self, frame, show_box=False):
        frame[self._bbox['top']:self._bbox['bottom'], self._bbox['left']: self._bbox['right'], :] = self._colors['bkg']

        # volume wedge
        volume = self._note.amplitude
        if volume is not None and volume > 0 and self._note.pushed:
            # x_right = self
            volume_pts = np.array([self._volume_left,
                                  self._volume_left*(1.0-volume) + self._volume_right_up * volume,
                                   self._volume_left*(1.0-volume) + self._volume_right_down*volume]) * 2.** PRECISION_BITS
            cv2.fillPoly(frame, [volume_pts.astype(np.int32)], self._colors['volume'],lineType=cv2.LINE_AA,shift=PRECISION_BITS)

        # rects
        for rect in self._rects:
            ((i0, i1), (j0, j1)) = rect['coords']

            frame[i0:i1, j0:j1, :] = rect['color']

        # lines
        for line_info in self._lines:
            color, thickness, closed = line_info
            cv2.polylines(frame, self._lines[line_info], closed, color, thickness=int(thickness), lineType=cv2.LINE_AA,
                          shift=PRECISION_BITS)

        # text (slowest?)

        cv2.putText(frame, LAYOUT['title']['txt'], self._title_pos, LAYOUT['title']['font'],
                    self._title_font_scale, self._colors['text'], self._title_thickness, cv2.LINE_AA)

        cv2.putText(frame, "ppp", self._ppp_pos, LAYOUT['dynamics']['font'],
                    self._dynamics_font_scale, self._colors['text'], self._dynamics_thickness, cv2.LINE_AA)
        cv2.putText(frame, "fff", self._fff_pos, LAYOUT['dynamics']['font'],
                    self._dynamics_font_scale, self._colors['text'], self._dynamics_thickness, cv2.LINE_AA)

        if show_box:
            cv2.polylines(frame, self._note_box_coords, True, self._colors['guide_box'], thickness=2,
                          lineType=cv2.LINE_AA)

        self._note.draw(frame)

    def mouse(self, event, x, y, flags, param):
        """
        Set note position to mouse coordinates only if button is down or mouse is in area
        """

        self._mouse_pos = x, y
        in_note_area = (self._note_bbox['top'] <= y < self._note_bbox['bottom'] and
                        self._note_bbox['left'] <= x < self._note_bbox['right'])

        if event == cv2.EVENT_LBUTTONDOWN:
            self._button_down = True
            if in_note_area:
                self._note.set_pos(x, y)
                self._note.push()

        elif event == cv2.EVENT_LBUTTONUP:
            self._button_down = False
            self._note.unpush()
        else:
            if in_note_area or self._button_down:
                self._note.set_pos(x, y)


def test_staff_drawing():
    window_size = (590, 600)
    s = Staff({'top': 10, 'bottom': window_size[1] - 10, 'left': 10, 'right': window_size[0] - 10})
    blank = np.zeros((window_size[1], window_size[0], 4), dtype=np.uint8)
    win_name = "Staff test"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win_name, s.mouse)

    t_start = time.perf_counter()

    n_frames = 0

    while True:
        frame = blank.copy()
        s.draw(frame, show_box=True)

        cv2.imshow(win_name, frame)
        k = cv2.waitKey(1)
        if k & 0xff == ord('q'):
            break
        n_frames += 1
        now = time.perf_counter()
        if now - t_start > 2:
            print("FPS:  %.3f" % (n_frames / (now - t_start),))
            t_start = now
            n_frames = 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_staff_drawing()
