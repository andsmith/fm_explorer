"""
Gui code for main "instrument", the score & note display.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from util import add_text, get_ellipse_axes, eval_ellipse, floats_to_pixels, make_image_from_alpha, in_unit_box
from clefs import make_bass, make_treble


def _line_wrap(img, p1, p2, color, **kwargs):
    rv = cv2.line(img, p1, p2, color, **kwargs)
    return rv


class StaffAnimator(object):
    """Musical display"""
    COLORS = {'color_light': (245, 235, 229),
              'color_dark': (0, 0, 0),
              'guidebox_color': (64, 255, 64),
              'volume_color': (32, 200, 32)}
    NOTE_ECCENTRICITY = 0.75
    NOTE_ROTATION_DEG = 15.0
    LAYOUT = {'staff_v_span': [.25, .7],  # all dims relative to bbox  (unit)
              'staff_h_span': [.1, .9],
              'clef_indent_spaces': 1.75,
              'title': {'txt': 'Theremin',
                        'pos': [.5, .045],
                        'font': cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                        'font_scale_mult': 1},
              'ledger_line_length_mult': 1.5}  # x width of note

    def __init__(self, bbox, middle_c_hz=261.625565):
        """
        
        :param bbox:  dict with 'top','bottom','left','right'
        :param shape: of main app window
        :param middle_c_hz:  for converting from xy to hz/amp
        """
        self._bbox = bbox
        self._width, self._height = self._bbox['right'] - self._bbox['left'], self._bbox['bottom'] - self._bbox['top']
        self._size = np.array((self._width, self._height))
        self._middle_c_hz = middle_c_hz
        self._note_ecc = StaffAnimator.NOTE_ECCENTRICITY
        self._note_phi_deg = StaffAnimator.NOTE_ROTATION_DEG
        self._gen_drawing_data()

        self._current_note_data = None
    def _gen_drawing_data(self, ):
        """"""
        min_line_width = 2
        staff_top_y = self._bbox['top'] + int(self._height * StaffAnimator.LAYOUT['staff_v_span'][0])
        staff_bottom_y = self._bbox['top'] + int(self._height * StaffAnimator.LAYOUT['staff_v_span'][1])
        staff_left_x = self._bbox['left'] + int(self._width * StaffAnimator.LAYOUT['staff_h_span'][0])
        staff_right_x = self._bbox['left'] + int(self._width * StaffAnimator.LAYOUT['staff_h_span'][1])

        staff_height = staff_bottom_y - staff_top_y
        staff_width = staff_right_x - staff_left_x
        spacing = staff_height / 10
        note_stem_width = spacing / 15.0
        staff_line_width = int(note_stem_width * 1.5)
        staff_line_width = min_line_width if staff_line_width < min_line_width else staff_line_width
        staff_line_half_width = int(staff_line_width / 2)

        # crescendo
        wedge_left_x = self._bbox['left'] + staff_left_x - staff_line_width + spacing * 3.5
        wedge_right_x = self._bbox['left'] + staff_right_x - spacing * 3
        wedge_y = staff_bottom_y + int(spacing * 4)
        wedge_right_y = wedge_y - spacing * .5, \
                        wedge_y + spacing * .5

        h = int(spacing / 2. - staff_line_width / 2.)  # note half-height
        phi_rad = np.deg2rad(self._note_phi_deg)

        # now define note shape stuff
        n_samples = 30  # int(h * 4) if int(h * 4) > 100 else 100
        t = np.linspace(0, np.pi * 2, n_samples)
        note_shape_bounds = eval_ellipse(t, self._note_ecc, phi_rad, h)
        error = np.max(note_shape_bounds, axis=0) + np.min(note_shape_bounds, axis=0)
        note_shape_bounds -= error  # re-center
        note_shape_bounds = np.int64(np.around(note_shape_bounds))
        note_width = np.max(note_shape_bounds[:, 0]) - np.min(note_shape_bounds[:, 0])

        # stem attachment points
        left_most = np.min(note_shape_bounds[:, 0])
        all_lefties = np.where(note_shape_bounds[:, 0] == left_most)[0]
        left_attachment_point = np.mean(note_shape_bounds[all_lefties, :], axis=0)

        right_most = np.max(note_shape_bounds[:, 0])
        all_righties = np.where(note_shape_bounds[:, 0] == right_most)[0]
        right_attachment_point = np.mean(note_shape_bounds[all_righties, :], axis=0)

        middle_c_y = ((staff_top_y + staff_bottom_y) / 2)
        treble_y = middle_c_y - spacing * 2.0  # G
        bass_y = middle_c_y + spacing * 2.0  # F
        clef_x = int(staff_left_x + StaffAnimator.LAYOUT['clef_indent_spaces'] * spacing)
        treble_pos = np.array((clef_x, treble_y)).reshape(1, -1)
        bass_pos = np.array((clef_x, bass_y)).reshape(1, -1)

        bass_coords = [(bass_pos + coords * spacing, is_closed) for coords, is_closed in zip(*make_bass())]
        treble_coords = [(treble_pos + coords * spacing, is_closed) for coords, is_closed in zip(*make_treble())]

        self._dims = {'top_y': staff_top_y,
                      'bottom_y': staff_bottom_y,
                      'middle_c_y': middle_c_y,
                      'right_x': staff_right_x,
                      'left_x': staff_left_x,
                      'height': staff_height,
                      'width': staff_width,
                      'spacing': spacing,
                      'phi': phi_rad,
                      'ecc': self._note_ecc,
                      'ledger_line_length': int(StaffAnimator.LAYOUT['ledger_line_length_mult'] * note_width),
                      'line_width': staff_line_width,
                      'line_half_width': staff_line_half_width,
                      'bass_clef_pos': bass_pos,
                      'treble_clef_pos': treble_pos,
                      'clef_coords': treble_coords + bass_coords,
                      'note_bbox': self._bbox,
                      'wedge_left_x': wedge_left_x,
                      'wedge_right_x': wedge_right_x,
                      'wedge_y': wedge_y,
                      'wedge_right_y': wedge_right_y,
                      'wedge_text_spacing': int(spacing * .8),
                      'h': h,
                      'stem_width': note_stem_width,
                      'notehead_shape': note_shape_bounds,
                      'stem_attachment_point_down': left_attachment_point,
                      'stem_attachment_point_up': right_attachment_point, }
    def _make_staff(self):
        pass

    def draw_frame(self, show_guidebox=False):
        # handy
        dims = self._dims
        spacing = self._dims['spacing']
        bright = 254

        # first, greyscale, then convert to color
        f = np.zeros(self._size[::-1], dtype=np.uint8)

        def _draw_line(x, y, d, half_width):
            if d == 'v':
                f[y[0]:y[1], x - half_width: x + half_width] = bright
            elif d == 'h':
                f[y - half_width: y + half_width, x[0]:x[1]] = bright

        # draw staff lines
        staff_h_line_x = [dims['left_x'], dims['right_x']]
        staff_h_line_y = np.linspace(dims['top_y'], dims['bottom_y'], 11).astype(np.int64).tolist()
        staff_h_line_y = np.array(staff_h_line_y[:5] + staff_h_line_y[6:])
        staff_v_left_double_x = dims['left_x']
        staff_v_left_double_y = [dims['top_y'] - dims['line_half_width'], dims['bottom_y'] + dims['line_half_width']]
        staff_v_left_single_x = dims['left_x'] + dims['line_width'] * 4
        staff_v_left_single_y = [dims['top_y'] - dims['line_half_width'], dims['bottom_y'] + dims['line_half_width']]
        for y in staff_h_line_y:
            _draw_line(staff_h_line_x, y, 'h', dims['line_half_width'])
        _draw_line(staff_v_left_double_x, staff_v_left_double_y, 'v', dims['line_width'])
        _draw_line(staff_v_left_single_x, staff_v_left_single_y, 'v', dims['line_half_width'])
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 1 + int(spacing / 40)
        self._draw_clefs(f)

        # draw crescendo wedge
        wedge_left = (int(dims['wedge_left_x']), int(dims['wedge_y']))
        wedge_right = (int(dims['wedge_right_x']), int(dims['wedge_y']))
        wedge_right_up = (int(dims['wedge_right_x']), int(dims['wedge_right_y'][0]))
        wedge_right_down = (int(dims['wedge_right_x']), int(dims['wedge_right_y'][1]))

        cv2.line(f, wedge_left, wedge_right_up, bright, thickness=dims['line_half_width'], lineType=cv2.LINE_AA)
        cv2.line(f, wedge_left, wedge_right_down, bright, thickness=dims['line_half_width'], lineType=cv2.LINE_AA)
        left_pos = (wedge_left[0] - dims['wedge_text_spacing'], wedge_left[1])
        right_pos = (wedge_right[0] + dims['wedge_text_spacing'], wedge_right[1])
        add_text(f, "ppp", left_pos, font=font, font_scale=font_scale / 1.5, h_align='right', v_align='middle')
        add_text(f, "fff", right_pos, font=font, font_scale=font_scale / 1.5, v_align='middle')

        # draw title

        title_pos = np.int32([StaffAnimator.LAYOUT['title']['pos'][0] * self._width,
                              StaffAnimator.LAYOUT['title']['pos'][1] * self._height])

        add_text(f, StaffAnimator.LAYOUT['title']['txt'], title_pos,
                 font=StaffAnimator.LAYOUT['title']['font'],
                 font_scale=StaffAnimator.LAYOUT['title']['font_scale_mult'] * font_scale,
                 v_align='top',
                 h_align='middle')

        # convert to color
        img = make_image_from_alpha(f,
                                    color_high=StaffAnimator.COLORS['color_dark'],
                                    color_low=StaffAnimator.COLORS['color_light'])
        # draw things in color here.
        if show_guidebox:
            gb_thickness = np.max([3, int(img.shape[0] / 100)])
            gb_color = StaffAnimator.COLORS['guidebox_color']
            guide_span_y = np.array([dims['note_bbox']['top'], dims['note_bbox']['bottom']], dtype=np.int64)
            guide_span_x = np.array([dims['note_bbox']['left'], dims['note_bbox']['right']], dtype=np.int64)
            pa1 = guide_span_x[0] + gb_thickness / 2, guide_span_y[0] + gb_thickness / 2
            pa2 = guide_span_x[0] + gb_thickness / 2, guide_span_y[1] - gb_thickness / 2
            pb1 = guide_span_x[1] - gb_thickness / 2, guide_span_y[0] + gb_thickness / 2
            pb2 = guide_span_x[1] - gb_thickness / 2, guide_span_y[1] - gb_thickness / 2
            guide_points = np.array([[pa1, pa2, pb2, pb1]], dtype=np.int32)
            # make new, contiguous array with copy
            img = cv2.polylines(img.copy(), guide_points, True, gb_color, thickness=gb_thickness, lineType=cv2.LINE_AA)
        return img, dims

    def _draw_clefs(self, img):
        COLOR = 254  # placeholder
        PRECISION_BITS = 6
        for coords, is_closed in self._dims['clef_coords']:
            coords = np.int32([coords*(2**PRECISION_BITS)])
            if is_closed:
                cv2.fillPoly(img, coords, COLOR, lineType=cv2.LINE_AA,shift=PRECISION_BITS)
            else:
                cv2.polylines(img, coords, is_closed, COLOR,
                              thickness=int(self._dims['stem_width']), lineType=cv2.LINE_AA, shift=PRECISION_BITS)
    '''

    def mouse(self, event, note_x, note_y, flags, param):
        """
        Determine frequency & amplitude, etc. from relative position within note area

        Note, center rel_pos (0.5, 0.5), corresponds to middle-C, which may not
        be in the center of the note_area vertically

        :param rel_pos: x, y on staff image, both in [0.0, 1.0]
        :return: data with musical info
        """
        box = self._dims['note_bbox']
        #rel_pos = np.array((box['left']-x, box['top']-y))
        
        staff_center = self._dims['middle_c_y']
        staff_dist = 2 * (staff_center - box['top']) * rel_pos[1]
        note_y = int(box['top'] + staff_dist)
        note_x = box['left'] + rel_pos[0] * (box['right'] - box['left'])

        pos = note_x, note_y

        step_distance = (pos[1] - self._dims['middle_c_y']) / self._dims['spacing']
        n_ledger_lines = int(np.ceil(np.abs(step_distance) - 5.5))
        if np.abs(step_distance) < 5.5:
            n_ledger_lines = 0

        show_middle_ledger_line = self._dims['middle_c_y'] - self._dims['spacing'] < \
                                  pos[1] < \
                                  self._dims['middle_c_y'] + self._dims['spacing']

        amplitude = (pos[0] - box['left']) / (box['right'] - box['left'])

        freq = 2. ** (step_distance * 2. / 12.)

        self._current_note_data = {'pos': pos,
                                   'show_middle_ledger_line': show_middle_ledger_line,
                                   'n_ledger_lines': n_ledger_lines,
                                   'step_distance': step_distance,
                                   'freq_hz': freq,
                                   'unscaled_amp': amplitude}
        return self._current_note_data

    def draw_note_data(self, img, note_data, show_volume=True, valid=True):
        """
        Draw note to image. (i.e. on a staff)
        :param img: target
        :param pos: row, column for center of notehead
        :param note_shape_data: return value of make_note_shape_data, for drawing
        :return: img with note on it.
        """
        dims = self._dims
        step_distance = note_data['step_distance']
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
        img = img.copy()  # np.ascontiguousarray(img, dtype=np.uint8)

        # volume
        if show_volume:
            shift_x = dims['line_width']  # move to the right a bit to not overlap wedge
            volume_left_y = dims['wedge_y']
            volume_left_x = dims['wedge_left_x'] + shift_x
            vol_frac = note_data['unscaled_amp']

            wedge_x_dist = float(dims['wedge_right_x'] - dims['wedge_left_x'] - shift_x)
            wedge_y_dist = float(dims['wedge_right_y'][1] - dims['wedge_right_y'][0] - dims['line_width'])

            volume_x_dist = int(vol_frac * wedge_x_dist)
            volume_y_dist = int(vol_frac * wedge_y_dist)

            volume_right_x = volume_left_x + volume_x_dist
            volume_right_y = volume_left_y - int(volume_y_dist / 2), \
                             volume_left_y + int(volume_y_dist / 2)

            volume_xy = np.array([(volume_left_x, volume_left_y),
                                  (volume_right_x, volume_right_y[0]),
                                  (volume_right_x, volume_right_y[1])])
            volume_xy = np.int32([volume_xy])
            cv2.fillPoly(img, volume_xy, StaffAnimator.COLORS['volume_color'], lineType=cv2.LINE_AA)

        if note_data['show_middle_ledger_line']:
            middle_ledger_y_span = int(dims['middle_c_y']) - dims['line_half_width'], \
                                   int(dims['middle_c_y']) + dims['line_half_width']
            middle_ledger_x_span = int(pos[0] - dims['ledger_line_length'] / 2), \
                                   int(pos[0] + dims['ledger_line_length'] / 2)
            img[middle_ledger_y_span[0]: middle_ledger_y_span[1],
            middle_ledger_x_span[0]: middle_ledger_x_span[1]] = 0

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

        # head
        notehead = np.int32([notehead])

        if valid:
            cv2.fillPoly(img, notehead, (0, 0, 0), lineType=cv2.LINE_AA)
        else:
            cv2.polylines(img, notehead, True, (0, 0, 0),
                          thickness=int(dims['stem_width']), lineType=cv2.LINE_AA)

        return img
'''

def test_staff_drawing():
    s = StaffAnimator({'top': 10, 'bottom': 990, 'left': 10, 'right': 890})
    img, dims = s.draw_frame()

    # note_data = s.get_note_data((0.5, 0.5))
    # img = s.draw_note_data(img, note_data=note_data)
    plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    test_staff_drawing()
