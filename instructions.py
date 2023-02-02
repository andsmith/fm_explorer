HELP_TEXT = ["FM Explorer - find interesting parameters for FM synthesis                     by Andrew T. Smith (2023)",
             '------------------------------------------------------------------',
             ' ',
             '  FM(t) = A(t) * sin [2 * pi * Fc(t) * t + D(t) * sin(2 * pi * Fm(t) * t)]',
             '          ------------------   ------------------',
             '           Carrier                           Modulation',
             ' ',
             '    * A(t) = carrier amplitude, Fc(t) = carrier frequency',
             '    * D(t) = modulation depth,  Fm(t) = modulation frequency',
             ' ',
             ' Instructions:',
             '     -->  Move mouse to change params.',
             '     -->  Hit space to start / stop sound.',
             '     -->  Hit M or C to change modes (Modulation / Carrier)',
             ' ',
             '  All hotkeys:',
             '      h - Help screen on / off.',
             "     ' ' -  Play Sound / Stop Sound",
             ' ',
             "     x  -  Scale modulation freq with carrier to preserve timbre (on/off).",
             "     c, m - Toggle mode (adjusting Modulation / Carrier)",
             "     s  -  save current modulation params",
             "     q  -  Quit                           ",
             ' ' ,
             "     0  -  adjust vertical axis range up (some axes are locked)",
             "     p  -  adjust vertical axis range down",
             ' ',
             "     ]  -  adjust horizontal axis range up",
             "     [  -  adjust horizontal axis range down",
             ' ',
             "     .  -  FFT window smaller",
             "     ,  -  FFT window larger"
             ]
