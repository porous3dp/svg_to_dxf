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


# Hack for most common use-case:
#   If no output filename is specified, but a single arg is,
#   assume that the arg is the input file.
if len(sys.argv) > 1 and ('--output' not in sys.argv):
	fileN = sys.argv[1]
	#print("fileN = '%s'" % fileN)
	assert fileN.endswith('.svg')
	
	output_filename = os.path.splitext(fileN)[0] + '.dxf'
	sys.argv[1:1] = ['--output', output_filename]


# Use default units of "mm" (as per dxf_outlines.inx, line 14)
sys.argv[1:1] = ['--units', '25.4/96']


# Run the extension in standalone mode from the "dxf_outlines.py" file
# (see the `if __name__ == '__main__': ...` stuff there)
dxf_outlines.DxfOutlines().run()

