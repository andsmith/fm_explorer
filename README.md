# fm_explorer
Explore sounds created by FM synthesis!

See http://www.cs.cmu.edu/~music/icm-online/readings/fm-synthesis/index.html

## Instructions

0. Turn your volume down.
1. Run:

    > python explore.py

 2. * Use the mouse to adjust the carrier and modulation controls (freq/amp) of an FM Synthesizer.
    * Hit SPACE to stop/start sound (also, mouse-click on the note in theremin mode)
    * Hit C and T to select carrier/theremin modes.
    * Hit A to cycle through auto-tune modes (in theremin mode), featuring:
        * None (continuous freuqencies)
        * Chromatic 
        * Diatonic
        * Pentatonic

Help screen:

![FM Explorer - info splash](https://github.com/andsmith/fm_explorer/blob/main/screen_help.png)

Direct Carrier adjusting (& modulation):

![FM Explorer - carrier adjust](https://github.com/andsmith/fm_explorer/blob/main/screen_carrier.png)

Theremin mode:

![res_detect_user_pick](https://github.com/andsmith/fm_explorer/blob/main/screen_theremin.png)


Future work:
  * voice memory bank
  * hand-tracking control/modulation
  * Sound ON/OFF indicator.
  * attack sustain decay release envelope for mouse click tones in Theremin
  * Do more interesting (or "musical") parameters have integer or rational ratios with low denominators between the carrier and modulation frequencies?  These could be made "sticky" on the modulation grid?  Not sure what to do with modulation depth/amplitude though, i.e. which values produce more musical FM parameters...
