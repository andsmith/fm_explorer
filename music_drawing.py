"""
Gui code for main "instrument", the score & note display.
"""
import time
import logging
import numpy as np
import cv2
from music_drawing_util import add_text, eval_ellipse
from clef_drawing import make_bass, make_treble

from music_layout import DEFAULT_COLORS_BGRA, PRECISION_BITS, NOTE_ROTATION_DEG, NOTE_ECCENTRICITY

from staff import Staff, Note


class StaffAnimator(object):
    """
    Musical display
    Update an image (frame) in real time based on mouse position.
    Translate mouse position to musical information about pitch and dynamics.
    """

    def __init__(self, bbox, a_hz=440.0, colors=None):
        """
        
        :param bbox:  dict with 'top','bottom','left','right'
        :param a_hz:  for converting from xy to hz/amp
        :param colors: dict with keys in StaffAnimator.DEFAULT_COLORS_BGR, values (b,g,r) uint8
        """
        self._colors = colors if colors is not None else DEFAULT_COLORS_BGRA
        self._bbox = bbox
        self._staff = Staff(self._bbox, colors = self._colors)
        self._note = Note(self._bbox, colors=self._colors)

        self._middle_c_hz = a_hz * 2 ** (-9 / 12)
        logging.info("Setting middle-c:  %.3f hz" % (self._middle_c_hz,))

        self._set_components()

        self._mouse_pos = None
        self._current_note_data = None  # derived from mouse position

    def _set_components(self):
        """
        Determine dimensions / layout of everything that doesn't change.

        lines are defined by endpoints & thickness

        line thicknesses are integer, coordinates are floats, except where needed to prevent aliasing, etc.

        """

        # crescendo wedge
        wedge_left_x = self._staff_left_x + self._space * 7  # todo:  move relative dimensions to def at top
        wedge_right_x = self._staff_right_x - self._space * 5
        wedge_y_center = self._staff_bottom_y + int(self._space * 4)
        wedge_right_ys = wedge_y_center - self._space * .5, \
                         wedge_y_center + self._space * .5
        wedge_left = wedge_left_x, wedge_y_center
        wedge_right_up = wedge_right_x, wedge_right_ys[0]
        wedge_right_down = wedge_right_x, wedge_right_ys[1]
        self._wedge = Component().add_polyline(np.array((wedge_right_up, wedge_left, wedge_right_down)),

                                               self._colors['lines'],
                                               thickness=self._note_stem_thickness,
                                               closed=False,
                                               filled=False
                                               )

        # bounding box of note area:
        self._note_bbox = {'top': np.max((self._staff_top_y,
                                          self._middle_c_y - self._space * (5 + layout['n_max_ledger_lines'][1]))),
                           'bottom': np.min((self._staff_top_y,
                                             self._middle_c_y - self._space * (5 + layout['n_max_ledger_lines'][1]))),
                           'left': wedge_left_x, 'right': wedge_right_x}

        # note shape
        t = np.linspace(0, np.pi * 2, 30)  # increase for better note shape
        phi_rad = np.deg2rad(StaffAnimator.NOTE_ROTATION_DEG)
        h = self._space / 2. - self._staff_line_thickness / 2.  # note half-height
        note_shape = eval_ellipse(t, StaffAnimator.NOTE_ECCENTRICITY, phi_rad, h)
        error = (np.max(note_shape, axis=0) + np.min(note_shape, axis=0)) / 2

        self._note_rel_coords = note_shape - error  # re-center
        self._note_width = np.max(note_shape[:, 0]) - np.min(note_shape[:, 0])

        # stem attachment points, right & left most point of note shape
        left_most = np.min(self._note_rel_coords[:, 0])
        all_lefties = np.where(self._note_rel_coords[:, 0] == left_most)[0]
        self._left_attach = np.mean(self._note_rel_coords[all_lefties, :], axis=0)
        right_most = np.max(self._note_rel_coords[:, 0])
        all_righties = np.where(self._note_rel_coords[:, 0] == right_most)[0]
        self._right_attach = np.mean(self._note_rel_coords[all_righties, :], axis=0)


    def _add_note_area_box(self, img, color):
        gb_thickness = np.max([3, int(img.shape[0] / 100)])
        guide_span_y = np.array([self._note_bbox['top'], self._note_bbox['bottom']], dtype=np.int64)
        guide_span_x = np.array([self._note_bbox['left'], self._note_bbox['right']], dtype=np.int64)
        pa1 = guide_span_x[0] + gb_thickness / 2, guide_span_y[0] + gb_thickness / 2
        pa2 = guide_span_x[0] + gb_thickness / 2, guide_span_y[1] - gb_thickness / 2
        pb1 = guide_span_x[1] - gb_thickness / 2, guide_span_y[0] + gb_thickness / 2
        pb2 = guide_span_x[1] - gb_thickness / 2, guide_span_y[1] - gb_thickness / 2
        guide_points = np.array([[pa1, pa2, pb2, pb1]])
        # make new, contiguous array with copy
        draw_lines(img, guide_points, color, True, gb_thickness

    def draw(self, frame, show_box=False):
        colors = {c: self._colors[c][:frame.shape[2]] for c in self._colors}  # bgra -> bgr if needed

        # draw staff lines
        for line in self._staff_h_lines:
            draw_lines(frame, [line], colors['lines'], thickness=self._staff_line_thickness, closed=False)

        def draw_list(items, color, thickness):
            for coords, closed in items:
                draw_lines(frame, coords, color, thickness, closed=closed, filled=False)

        draw_lines(frame, self._staff_heavy_v_line], colors[
                                                         'lines'], self._staff_line_thickness * 4, closed = False, filled = False)
        draw_lines(frame, [self._staff_light_v_line], colors['lines'], self._staff_line_thickness, closed=False,
        filled = False)

        # draw clefs
        draw_lines(frame, self._treble_clef_coords, colors['lines'], thickness=self._stem_thickness, closed=is_closed,
        filled = is_closed)
        draw_lines(frame, self._bass_clef_coords, colors['lines'], thickness=self._stem_thickness, closed=is_closed,

    filled = is_closed)

    # draw crescendo wedge
    draw_lines(frame, points, bright, False, dims['line_half_width'])

    left_pos = (wedge_left[0] - dims['wedge_text_spacing'], wedge_left[1])
    right_pos = (wedge_right[0] + dims['wedge_text_spacing'], wedge_right[1])
    add_text(frame, "ppp", left_pos, font=font, font_scale=font_scale / 1.5, h_align='right', v_align='middle')
    add_text(frame, "fff", right_pos, font=font, font_scale=font_scale / 1.5, v_align='middle')

    # draw title
    title_pos = np.int32([StaffAnimator.LAYOUT['title']['pos'][0] * self._width,
                          StaffAnimator.LAYOUT['title']['pos'][1] * self._height])

    add_text(frame, StaffAnimator.LAYOUT['title']['txt'], title_pos,


font = StaffAnimator.LAYOUT['title']['font'],
font_scale = StaffAnimator.LAYOUT['title']['font_scale_mult'] * font_scale,
v_align = 'top',
h_align = 'middle')

if show_guidebox:
    self._add_guidebox(img, colors['guide_box'])


def mouse(self, event, x, y, flags, param):
    """
    Determine frequency & amplitude, etc. from relative position within note area

    Note, center rel_pos (0.5, 0.5), corresponds to middle-C, which may not
    be in the center of the note_area vertically

    :param rel_pos: x, y on staff image, both in [0.0, 1.0]
    :return: data with musical info
    """

    self._mouse_pos = x, y
    self._current_note_data = self._note_info_from_pos(x, y)


def _note_info_from_pos(self, note_x, note_y):
    halfsteps_to_middle_c = (note_y - self._dims['middle_c_y']) / self._dims['spacing']
    n_ledger_lines = int(np.ceil(np.abs(halfsteps_to_middle_c) - 5.5))
    if np.abs(halfsteps_to_middle_c) < 5.5:
        n_ledger_lines = 0

    show_middle_ledger_line = self._dims['middle_c_y'] - self._dims['spacing'] < \
                              note_y < \
                              self._dims['middle_c_y'] + self._dims['spacing']
    width = self._dims['wedge_right_x'] - self._dims['wedge_left_x']
    amplitude = np.clip((note_x - self._dims['wedge_left_x']) / width, 0, 1.0)
    freq = 2. ** (halfsteps_to_middle_c * 2. / 12.)
    print(amplitude)
    return {'pos': (note_x, note_y),
            'show_middle_ledger_line': show_middle_ledger_line,
            'n_ledger_lines': n_ledger_lines,
            'halfsteps_to_middle_c': halfsteps_to_middle_c,
            'freq_hz': freq,
            'amplitude': amplitude}


def draw_note(self, img, show_volume=True, valid=True):
    """
    Draw note to image. (i.e. on a staff)
    :param img: target
    :param pos: row, column for center of notehead
    :param note_shape_data: return value of make_note_shape_data, for drawing
    :return: img with note on it.
    """
    if self._current_note_data is None:
        return
    colors = {c: self._colors[c][:img.shape[2]] for c in self._colors}  # bgra -> bgr if needed

    note_data = self._current_note_data
    dims = self._dims
    step_distance = note_data['halfsteps_to_middle_c']
    n_ledger_lines = note_data['n_ledger_lines']
    pos = np.array(note_data['pos'], dtype=np.int64)
    notehead = dims['notehead_shape'] + pos

    # calculate ledger line & note stem locations
    if n_ledger_lines > 0:
        ledger_y_pos_offsets = np.int64(np.arange(6, 6 + n_ledger_lines) * dims['spacing'])
    else:
        ledger_y_pos_offsets = np.array([], dtype=(np.int64))
    ledger_x_span = pos[0] - int(dims['ledger_line_length'] / 2), \
                    pos[0] + int(dims['ledger_line_length'] / 2)
    if step_distance < 0:
        ledger_y_pos = dims['middle_c_y'] - ledger_y_pos_offsets
        stem_attach = dims['stem_attachment_point_down'] + pos
        stem_x_span = np.array([stem_attach[0], stem_attach[0] + dims['stem_width']]).astype(np.int64)
    else:
        ledger_y_pos = dims['middle_c_y'] + ledger_y_pos_offsets
        stem_attach = dims['stem_attachment_point_up'] + pos
        stem_x_span = np.array([stem_attach[0] - dims['stem_width'], stem_attach[0]]).astype(np.int64)
    ledger_y_pos = np.int64(ledger_y_pos)

    # volume
    if show_volume:
        shift_x = dims['line_width']  # move to the right a bit to not overlap wedge
        volume_left_y = dims['wedge_y']
        volume_left_x = dims['wedge_left_x']
        vol_frac = note_data['amplitude']

        wedge_x_dist = float(dims['wedge_right_x'] - dims['wedge_left_x'] - shift_x)
        wedge_y_dist = float(dims['wedge_right_y'][1] - dims['wedge_right_y'][0] - dims['line_width'])

        volume_x_dist = int(vol_frac * wedge_x_dist)
        volume_y_dist = int(vol_frac * wedge_y_dist)

        volume_right_x = volume_left_x + volume_x_dist
        volume_right_y = volume_left_y - int(volume_y_dist / 2), \
                         volume_left_y + int(volume_y_dist / 2)

        volume_xy = np.array([(volume_left_x, volume_left_y),
                              (volume_right_x, volume_right_y[0]),
                              (volume_right_x, volume_right_y[1])]) * (2. ** StaffAnimator.PRECISION_BITS)
        volume_xy = np.int32([volume_xy])
        cv2.fillPoly(img, volume_xy, colors['volume'], lineType=cv2.LINE_AA,
                     shift=StaffAnimator.PRECISION_BITS)

    if note_data['show_middle_ledger_line']:
        middle_ledger_y_span = int(dims['middle_c_y']) - dims['line_half_width'], \
                               int(dims['middle_c_y']) + dims['line_half_width']
        middle_ledger_x_span = int(pos[0] - dims['ledger_line_length'] / 2), \
                               int(pos[0] + dims['ledger_line_length'] / 2)
        img[middle_ledger_y_span[0]: middle_ledger_y_span[1],
        middle_ledger_x_span[0]: middle_ledger_x_span[1]] = 128

    # stem length

    if np.abs(step_distance) > 6.5:
        assert n_ledger_lines >= 2, "Bad ledger line calc!"
        if step_distance > 0:
            stem_length = int(np.abs(pos[1] - dims['middle_c_y'] - dims['spacing'] * 3.))
        else:
            stem_length = int(np.abs(dims['middle_c_y'] - pos[1] - dims['spacing'] * 3.))
        stem_length -= dims['line_half_width']

    else:
        stem_length = int(3.5 * dims['spacing'])

    # draw ledger lines
    for y in ledger_y_pos:
        y_span = y - dims['line_half_width'], \
                 y + dims['line_half_width']

        img[y_span[0]:y_span[1], ledger_x_span[0]: ledger_x_span[1], :] = 0
    if step_distance > 0:
        # stem up
        stem_y_span = np.array([stem_attach[1] - stem_length, stem_attach[1]]).astype(np.int64)
    else:
        # stem_down
        stem_y_span = np.array([stem_attach[1], stem_attach[1] + stem_length]).astype(np.int64)

    # adjust stem for non-solid notes
    if not valid:
        if step_distance < 0:
            stem_x_span -= dims['line_half_width']
        else:
            stem_x_span += dims['line_half_width']

    # no ledger for "middle" c
    middle_band = .25
    if not (dims['middle_c_y'] - dims['spacing'] * middle_band
            < pos[1]
            < dims['middle_c_y'] + dims['spacing'] * middle_band):
        img[stem_y_span[0]:stem_y_span[1], stem_x_span[0]: stem_x_span[1], :] = 0

    # notehead
    if valid:
        notehead = np.int32([notehead * (2. ** StaffAnimator.PRECISION_BITS)])
        cv2.fillPoly(img, notehead, colors['note'], lineType=cv2.LINE_AA, shift=StaffAnimator.PRECISION_BITS)
    else:
        draw_lines(img, notehead, colors['note'], self._stem_thickness, closed=True, filled=False)


def test_staff_drawing():
    window_size = (1000, 900)
    s = StaffAnimator({'top': 10, 'bottom': window_size[1] - 10, 'left': 10, 'right': window_size[0] - 10})
    blank = np.zeros((window_size[1], window_size[0], 3), dtype=np.uint8)
    win_name = "Staff test"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win_name, s.mouse)
    frame = blank.copy()
    s.draw(frame, show_box=False)
    return

    n_frames = 0
    t_start = time.perf_counter()
    while True:
        frame = blank.copy()
        s.draw(frame, show_guidebox=False)
        s.draw_note(frame)

        cv2.imshow(win_name, frame)
        k = cv2.waitKey(1)
        if k & 0xff == ord('q'):
            break
        n_frames += 1
        now = time.perf_counter()
        if now - t_start > 2:
            print("FPS:  %.3f" % (n_frames / (now - t_start),))
            t_start = now
            n_frames += 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_staff_drawing()
