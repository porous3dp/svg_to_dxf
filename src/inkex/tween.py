# coding=utf-8
#
# Copyright (C) 2005 Aaron Spike, aaron@ekips.org
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
#

import math
from .utils import X, Y

def interpcoord(coord_a, coord_b, time):
    """Interpolate single coordinate by the amount of time"""
    return coord_a + ((coord_b - coord_a) * time)


def interppoints(point1, point2, time):
    """Interpolate coordinate points by amount of time"""
    return [interpcoord(point1[X], point2[X], time), interpcoord(point1[Y], point2[Y], time)]


def tweenstylefloat(prop, start, end, time):
    sp = float(start[prop])
    ep = float(end[prop])
    return str(sp + (time * (ep - sp)))


def tweenstyleunit(svg, prop, start, end, time):  # moved here so we can call 'unittouu'
    scale = svg.unittouu('1px')
    sp = svg.unittouu(start.get(prop, '1px')) / scale
    ep = svg.unittouu(end.get(prop, '1px')) / scale
    return str(sp + (time * (ep - sp)))


def tweenstylecolor(prop, start, end, time):
    sr, sg, sb = parsecolor(start[prop])
    er, eg, eb = parsecolor(end[prop])
    return '#%s%s%s' % (tweenhex(time, sr, er), tweenhex(time, sg, eg), tweenhex(time, sb, eb))


def tweenhex(time, s, e):
    s = float(int(s, 16))
    e = float(int(e, 16))
    retval = hex(int(math.floor(s + (time * (e - s)))))[2:]
    if len(retval) == 1:
        retval = '0%s' % retval
    return retval


def parsecolor(c):
    r, g, b = '0', '0', '0'
    if c[:1] == '#':
        if len(c) == 4:
            r, g, b = c[1:2], c[2:3], c[3:4]
        elif len(c) == 7:
            r, g, b = c[1:3], c[3:5], c[5:7]
    return r, g, b
