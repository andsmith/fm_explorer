import cv2

DEFAULT_COLORS_BGRA = {'bkg': (229, 235, 245, 255),
                       'lines': (3, 7, 9, 255),
                       'text': (3, 7, 9, 255),
                       'notes': (3, 7, 9, 255),
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
