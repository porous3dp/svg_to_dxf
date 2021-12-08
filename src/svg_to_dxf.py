#!/usr/bin/python
#
# Wrapper around "dxf_outlines.py", giving the script a more user-friendly name to use.
# It also fiddles with the arguments list so that the standard settings don't need to be
# supplied each time.
#
# Author: Joshua Leung
# Note: This is NOT a part of the original Inkscape extension code. It's a wrapper around
#       that for our convenience.

import sys
import os

import dxf_outlines

# Use default units of "mm" (as per dxf_outlines.inx, line 14)
sys.argv[1:1] = ['--units', '25.4/96']

# Run the extension in standalone mode from the "dxf_outlines.py" file
# (see the `if __name__ == '__main__': ...` stuff there)
dxf_outlines.DxfOutlines().run()

