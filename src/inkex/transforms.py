# coding=utf-8
#
# Copyright (C) 2006 Jean-Francois Barraud, barraud@math.univ-lille1.fr
# Copyright (C) 2010 Alvin Penner, penner@vaxxine.com
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
# barraud@math.univ-lille1.fr
#
# This code defines several functions to make handling of transform
# attribute easier.
#
"""
Provide transformation parsing to extensions
"""

import re
import sys
from decimal import Decimal
from math import cos, radians, sin, sqrt, tan, fabs, atan2, hypot, pi, isnan

from .utils import strargs, KeyDict, PY3

try:
    from typing import overload, List, Tuple, Union, Optional  # pylint: disable=unused-import

    VectorLike = Union["ImmutableVector2d", Tuple[float, float]]  # pylint: disable=invalid-name
    BoundingIntervalArgs = Union['BoundingInterval', Tuple[float, float], float]  # pylint: disable=invalid-name
except ImportError:
    overload = lambda x: x

# All the names that get added to the inkex API itself.
__all__ = (
    'BoundingBox',
    'DirectedLineSegment',
    'ImmutableVector2d',
    'Transform',
    'Vector2d',
)

if PY3:
    unicode = str  # pylint: disable=redefined-builtin,invalid-name

# Old settings, supported because users click 'ok' without looking.
XAN = KeyDict({'l': 'left', 'r': 'right', 'm': 'center_x'})
YAN = KeyDict({'t': 'top', 'b': 'bottom', 'm': 'center_y'})
# Anchoring objects with given directions (see inx options)
CUSTOM_DIRECTION = {270: 'tb', 90: 'bt', 0: 'lr', 360: 'lr', 180: 'rl'}
DIRECTION = ['tb', 'bt', 'lr', 'rl', 'ro', 'ri']


class ImmutableVector2d(object):
    _x = 0
    _y = 0

    x = property(lambda self: self._x)
    y = property(lambda self: self._y)

    @overload
    def __init__(self):  # type: () -> None
        pass

    @overload
    def __init__(self, x, y):  # type: (float, float) -> None
        pass

    @overload
    def __init__(self, v):  # type: (VectorLike) -> None
        pass

    def __init__(self, *args):
        x, y = self._parse(args)
        self._x, self._y = float(x), float(y)

    @staticmethod
    def _parse(args):
        if not args:
            return 0.0, 0.0
        if len(args) == 1:
            point = args[0]
            if isinstance(point, ImmutableVector2d):
                return point.x, point.y
            elif isinstance(point, (tuple, list)) and len(point) == 2:
                return point
            elif isinstance(point, str) and point.count(',') == 1:
                x, y = point.split(',')
                return x, y
        elif len(args) == 2:
            x, y = args
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                return x, y
        raise ValueError("Vector2d can't be constructed from {}".format(repr(args)))

    def __add__(self, other):  # type: (VectorLike) -> Vector2d
        other = Vector2d(other)
        return Vector2d(self.x + other.x, self.y + other.y)

    def __radd__(self, other):  # type: (VectorLike) -> Vector2d
        other = Vector2d(other)
        return Vector2d(self.x + other.x, self.y + other.y)

    def __sub__(self, other):  # type: (VectorLike) -> Vector2d
        other = Vector2d(other)
        return Vector2d(self.x - other.x, self.y - other.y)

    def __rsub__(self, other):  # type: (VectorLike) -> Vector2d
        other = Vector2d(other)
        return Vector2d(-self.x + other.x, -self.y + other.y)

    def __neg__(self):  # type: () -> Vector2d
        return Vector2d(-self.x, -self.y)

    def __pos__(self):  # type: () -> Vector2d
        return Vector2d(self.x, self.y)

    def __floordiv__(self, factor):  # type: (float) -> Vector2d
        return Vector2d(self.x / float(factor), self.y / float(factor))

    def __truediv__(self, factor):  # type: (float) -> Vector2d
        return Vector2d(self.x / float(factor), self.y / float(factor))

    def __div__(self, factor):  # type: (float) -> Vector2d
        return Vector2d(self.x / float(factor), self.y / float(factor))

    def __mul__(self, factor):  # type: (float) -> Vector2d
        return Vector2d(self.x * factor, self.y * factor)

    def __abs__(self):
        return self.length

    def __rmul__(self, factor):  # type: (float) -> VectorLike
        return Vector2d(self.x * factor, self.y * factor)

    def __repr__(self):
        return "Vector2d({:.6g}, {:.6g})".format(self.x, self.y)

    def __str__(self):
        return "{:.6g}, {:.6g}".format(self.x, self.y)

    def __iter__(self):
        yield self.x
        yield self.y

    def __len__(self):
        return 2

    def __getitem__(self, item):
        return (self.x, self.y)[item]

    def to_tuple(self):
        return self.x, self.y

    def dot(self, other):  # type: (VectorLike) -> float
        other = Vector2d(other)
        return self.x * other.x + self.y * other.y

    def is_close(self, other, rtol=1e-5, atol=1e-8
                 ):  # type: (Union[VectorLike,Tuple[float,float]], Optional[float], Optional[float]) -> float
        other = Vector2d(other)
        delta = (self - other).length
        return delta < (atol + rtol * other.length)

    @property
    def length(self):  # type: () -> float
        return sqrt(fabs(self.dot(self)))


class Vector2d(ImmutableVector2d):
    """
    Represents an element of 2-dimensional Euclidean space
    """

    @ImmutableVector2d.x.setter
    def x(self, value):
        self._x = float(value)

    @ImmutableVector2d.y.setter
    def y(self, value):
        self._y = float(value)

    def __iadd__(self, other):  # type: (VectorLike) -> VectorLike
        other = Vector2d(other)
        self.x += other.x
        self.y += other.y
        return self

    def __isub__(self, other):  # type: (VectorLike) -> VectorLike
        other = Vector2d(other)
        self.x -= other.x
        self.y -= other.y
        return self

    def __imul__(self, factor):  # type: (float) -> VectorLike
        self.x *= factor
        self.y *= factor
        return self

    def __idiv__(self, factor):  # type: (float) -> VectorLike
        self.x /= factor
        self.y /= factor
        return self

    def __itruediv__(self, factor):  # type: (float) -> VectorLike
        self.x /= factor
        self.y /= factor
        return self

    def __ifloordiv__(self, factor):  # type: (float) -> VectorLike
        self.x /= factor
        self.y /= factor
        return self

    @overload
    def assign(self, x, y):  # type: (float, float) -> None
        pass

    @overload
    def assign(self, other):  # type: (VectorLike) -> None
        pass

    def assign(self, *args):
        self.x, self.y = Vector2d(*args)



class Transform(object):
    """A transformation object which will always reduce to a matrix and can
    then be used in combination with other transformations for reducing
    finding a point and printing svg ready output.

    Use with svg transform attribute input:

      tr = Transform("scale(45, 32)")

    Use with triad matrix input (internal representation):

      tr = Transform(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))

    Use with hexad matrix input (i.e. svg matrix(...)):

      tr = Transform((1.0, 0.0, 0.0, 1.0, 0.0, 0.0))

    Once you have a transformation you can operate tr * tr to compose,
    any of the above inputs are also valid operators for composing.
    """
    TRM = re.compile(r'(translate|scale|rotate|skewX|skewY|matrix)\s*\(([^)]*)\)\s*,?')
    absolute_tolerance = 1e-5

    def __init__(self, matrix=None, callback=None, **extra):
        self.callback = None
        self.matrix = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        if matrix is not None:
            # We parse a given string as an svg transformation instruction
            if isinstance(matrix, (str, unicode)):
                for func, values in self.TRM.findall(matrix.strip()):
                    getattr(self, 'add_' + func.lower())(*strargs(values))
            elif isinstance(matrix, Transform):
                self.matrix = matrix.matrix
            elif not isinstance(matrix, (tuple, list)):
                raise ValueError("Invalid transform type: {}".format(type(matrix).__name__))
            elif len(matrix) == 2:
                self.matrix = tuple(matrix[0]), tuple(matrix[1])
            elif len(matrix) == 6:
                self.matrix = tuple(matrix[::2]), tuple(matrix[1::2])
            else:
                raise ValueError("Matrix '{}' is not a valid transformation matrix".format(matrix))

        self.add_kwargs(**extra)
        # Set callback last, so it doesn't kick off just setting up the internal value
        self.callback = callback

    # These provide quick access to the svg matrix:
    #
    # [ a, c, e ]
    # [ b, d, f ]
    #
    a = property(lambda self: self.matrix[0][0])  # pylint: disable=invalid-name
    b = property(lambda self: self.matrix[1][0])  # pylint: disable=invalid-name
    c = property(lambda self: self.matrix[0][1])  # pylint: disable=invalid-name
    d = property(lambda self: self.matrix[1][1])  # pylint: disable=invalid-name
    e = property(lambda self: self.matrix[0][2])  # pylint: disable=invalid-name
    f = property(lambda self: self.matrix[1][2])  # pylint: disable=invalid-name

    def __bool__(self):
        return not self.__eq__(Transform())

    __nonzero__ = __bool__

    def add_matrix(self, *args):
        """Add matrix in order they appear in the svg hexad"""
        self.__imul__(Transform(args))

    def add_kwargs(self, **kwargs):
        """Add translations, scales, rotations etc using key word arguments"""
        for key, value in reversed(list(kwargs.items())):
            func = getattr(self, 'add_' + key)
            if isinstance(value, tuple):
                func(*value)
            elif value is not None:
                func(value)

    @overload
    def add_translate(self, dr):  # type: (VectorLike) -> None
        pass

    @overload
    def add_translate(self, tr_x, tr_y=0.0):  # type: (float, Optional[float]) -> None
        pass

    def add_translate(self, *args):
        if len(args) == 1 and isinstance(args[0], (int, float)):
            tr_x, tr_y = args[0], 0.0
        else:
            tr_x, tr_y = Vector2d(*args)
        self.__imul__(((1.0, 0.0, tr_x), (0.0, 1.0, tr_y)))

    def add_scale(self, sc_x, sc_y=None):
        """Add scale to this transformation"""
        sc_y = sc_x if sc_y is None else sc_y
        self.__imul__(((sc_x, 0.0, 0.0), (0.0, sc_y, 0.0)))

    @overload
    def add_rotate(self, deg, center):  # type: (float, VectorLike) -> None
        pass

    @overload
    def add_rotate(self, deg, center_x, center_y):  # type: (float, float, float) -> None
        pass

    def add_rotate(self, deg, *args):
        """Add rotation to this transformation"""
        center_x, center_y = Vector2d(*args)
        _cos, _sin = cos(radians(deg)), sin(radians(deg))
        self.__imul__(((_cos, -_sin, center_x), (_sin, _cos, center_y)))
        self.__imul__(((1.0, 0.0, -center_x), (0.0, 1.0, -center_y)))

    def add_skewx(self, deg):
        """Add skew x to this transformation"""
        self.__imul__(((1.0, tan(radians(deg)), 0.0), (0.0, 1.0, 0.0)))

    def add_skewy(self, deg):
        """Add skew y to this transformation"""
        self.__imul__(((1.0, 0.0, 0.0), (tan(radians(deg)), 1.0, 0.0)))

    def to_hexad(self):
        """Returns the transform as a hexad matrix (used in svg)"""
        return (val for lst in zip(*self.matrix) for val in lst)

    def is_translate(self, exactly=False):
        """Returns True if this transformation is ONLY translate"""
        tol = self.absolute_tolerance if not exactly else 0.0
        return fabs(self.a - 1) <= tol and abs(self.d - 1) <= tol and fabs(self.b) <= tol and fabs(self.c) <= tol

    def is_scale(self, exactly=False):
        """Returns True if this transformation is ONLY scale"""
        tol = self.absolute_tolerance if not exactly else 0.0
        return (fabs(self.e) <= tol and fabs(self.f) <= tol and
                fabs(self.b) <= tol and fabs(self.c) <= tol)

    def is_rotate(self, exactly=False):
        """Returns True if this transformation is ONLY rotate"""
        tol = self.absolute_tolerance if not exactly else 0.0
        return self._is_URT(exactly=exactly) and \
               fabs(self.e) <= tol and fabs(self.f) <= tol and fabs(self.a ** 2 + self.b ** 2 - 1) <= tol

    def rotation_degrees(self):
        """Return the amount of rotation in this transform"""
        if not self._is_URT(exactly=False):
            raise ValueError("Rotation angle is undefined for non-uniformly scaled or skewed matrices")
        return atan2(self.b, self.a) * 180 / pi

    def __str__(self):
        """Format the given matrix into a string representation for svg"""
        hexad = tuple(self.to_hexad())
        if self.is_translate():
            if not self:
                return ""
            return "translate({:.6g}, {:.6g})".format(self.e, self.f)
        elif self.is_scale():
            return "scale({:.6g}, {:.6g})".format(self.a, self.d)
        elif self.is_rotate():
            return "rotate({:.6g})".format(self.rotation_degrees())
        return "matrix({})".format(" ".join(format(var, '.6g') for var in hexad))

    def __repr__(self):
        """String representation of this object"""
        return "{}((({}), ({})))".format(
            type(self).__name__,
            ', '.join(format(var, '.6g') for var in self.matrix[0]),
            ', '.join(format(var, '.6g') for var in self.matrix[1]))

    def __eq__(self, matrix):
        """Test if this transformation is equal to the given matrix"""
        return all(fabs(l - r) <= self.absolute_tolerance
                   for l, r in zip(self.to_hexad(), Transform(matrix).to_hexad()))

    def __mul__(self, matrix):
        """Combine this transform's internal matrix with the given matrix"""
        # Conform the input to a known quantity (and convert if needed)
        other = Transform(matrix)
        # Return a transformation as the combined result
        return Transform((
            self.a * other.a + self.c * other.b,
            self.b * other.a + self.d * other.b,
            self.a * other.c + self.c * other.d,
            self.b * other.c + self.d * other.d,
            self.a * other.e + self.c * other.f + self.e,
            self.b * other.e + self.d * other.f + self.f))

    def __imul__(self, matrix):
        """In place multiplication of transform matrices"""
        self.matrix = (self * matrix).matrix
        if self.callback is not None:
            self.callback(self)
        return self

    def __neg__(self):
        """Returns an inverted transformation"""
        det = (self.a * self.d) - (self.c * self.b)
        # invert the rotation/scaling part
        new_a = self.d / det
        new_d = self.a / det
        new_c = -self.c / det
        new_b = -self.b / det
        # invert the translational part
        new_e = -(new_a * self.e + new_c * self.f)
        new_f = -(new_b * self.e + new_d * self.f)
        return Transform((new_a, new_b, new_c, new_d, new_e, new_f))

    def apply_to_point(self, point):  # type: (VectorLike) -> Vector2d
        """Transform a tuple (X, Y)"""
        if isinstance(point, str):
            raise ValueError("Will not transform string '{}'".format(point))
        point = Vector2d(point)
        return Vector2d(self.a * point.x + self.c * point.y + self.e,
                        self.b * point.x + self.d * point.y + self.f)

    def _is_URT(self, exactly=False):
        """
        Checks that transformation can be decomposed into product of
        Uniform scale (U), Rotation around origin (R) and translation (T)

        :return: decomposition as U*R*T is possible
        """
        tol = self.absolute_tolerance if not exactly else 0.0
        return (fabs(self.a - self.d) <= tol) and (fabs(self.b + self.c) <= tol)


class BoundingInterval(object):  # pylint: disable=too-few-public-methods
    """A pair of numbers that represent the minimum and maximum values."""

    @overload
    def __init__(self, other):  # type: (BoundingInterval) -> None
        pass

    @overload
    def __init__(self, pair):  # type: (Tuple[float, float]) -> None
        pass

    @overload
    def __init__(self, value):  # type: (float) -> None
        pass

    @overload
    def __init__(self, x, y):  # type: (float, float) -> None
        pass

    def __init__(self, x, y=None):
        if y is not None:
            if isinstance(x, (int, float, Decimal)) and isinstance(y, (int, float, Decimal)):
                self.minimum = x
                self.maximum = y
            else:
                raise ValueError("Not a number for scaling: {} ({},{})"
                                 .format(str((x, y)), type(x).__name__, type(y).__name__))

        else:
            value = x
            if isinstance(value, BoundingInterval):
                self.minimum = value.minimum
                self.maximum = value.maximum
            elif isinstance(value, (tuple, list)) and len(value) == 2:
                self.minimum, self.maximum = min(value), max(value)
            elif isinstance(value, (int, float, Decimal)):
                self.minimum = self.maximum = value
            else:
                raise ValueError("Not a number for scaling: {} ({})"
                                 .format(str(value), type(value).__name__))

    def __bool__(self):
        return not (isnan(self.minimum) or isnan(self.maximum))

    __nonzero__ = __bool__

    def __neg__(self):
        return BoundingInterval((-self.maximum, -self.minimum))

    def __add__(self, other):
        new = BoundingInterval(self)
        if other is not None:
            new += other
        return new

    def __iadd__(self, other):
        if other is None:
            return
        other = BoundingInterval(other)
        self.minimum = min((self.minimum, other.minimum))
        self.maximum = max((self.maximum, other.maximum))
        return self

    def __radd__(self, other):
        if other is None:
            return BoundingInterval(self)
        return self + other

    def __mul__(self, other):
        new = BoundingInterval(self)
        if other is not None:
            new *= other
        return new

    def __imul__(self, other):
        self.minimum *= other
        self.maximum *= other
        return self

    def __iter__(self):
        yield self.minimum
        yield self.maximum

    def __eq__(self, other):
        return tuple(self) == tuple(BoundingInterval(other))

    def __contains__(self, value):
        return self.minimum <= value <= self.maximum

    def __repr__(self):
        return "BoundingInterval({}, {})".format(self.minimum, self.maximum)

    @property
    def center(self):
        """Pick the middle of the line"""
        return self.minimum + ((self.maximum - self.minimum) / 2)

    @property
    def size(self):
        """Return the size difference minimum and maximum"""
        return self.maximum - self.minimum


class BoundingBox(object):  # pylint: disable=too-few-public-methods
    """
    Some functions to compute a rough bbox of a given list of objects.

    BoundingBox(other)
    BoundingBox(x, y)
    BoundingBox((x1, x2), (y1, y2))
    """

    width = property(lambda self: self.x.size)
    height = property(lambda self: self.y.size)
    top = property(lambda self: self.y.minimum)
    left = property(lambda self: self.x.minimum)
    bottom = property(lambda self: self.y.maximum)
    right = property(lambda self: self.x.maximum)
    center_x = property(lambda self: self.x.center)
    center_y = property(lambda self: self.y.center)

    @overload
    def __init__(self, other):  # type: (BoundingBox) -> None
        pass

    @overload
    def __init__(self, x, y):  # type: (BoundingIntervalArgs, BoundingIntervalArgs) -> None
        pass

    def __init__(self, x, y=None):
        if y is None:
            if isinstance(x, BoundingBox):
                x, y = x.x, x.y
            else:
                raise ValueError("Not a number for scaling: {} ({})"
                                 .format(str(x), type(x).__name__))
        self.x = BoundingInterval(x)
        self.y = BoundingInterval(y)

    def __bool__(self):
        return bool(self.x) and bool(self.y)

    __nonzero__ = __bool__

    def __neg__(self):
        return BoundingBox(-self.x, -self.y)

    def __add__(self, other):
        new = BoundingBox(self)
        if other is not None:
            new += other
        return new

    def __iadd__(self, other):
        if other is None:
            return self
        other = BoundingBox(other)
        self.x += other.x
        self.y += other.y
        return self

    def __radd__(self, other):
        if other is not None:
            return self + other
        return self

    def __mul__(self, factor):
        new = BoundingBox(self)
        new *= factor
        return new

    def __imul__(self, factor):
        self.x *= factor
        self.y *= factor
        return self

    def __eq__(self, other):
        if isinstance(other, BoundingBox):
            return tuple(self) == tuple(other)
        return False

    def __iter__(self):
        yield self.x
        yield self.y

    @property
    def minimum(self):
        """Return the minimum x,y coords"""
        return Vector2d(self.x.minimum, self.y.minimum)

    @property
    def maximum(self):
        """Return the maximum x,y coords"""
        return Vector2d(self.x.maximum, self.y.maximum)

    def __repr__(self):
        return "BoundingBox({},{})".format(tuple(self.x), tuple(self.y))

    @property
    def center(self):
        """Returns the middle of the bounding box"""
        return Vector2d(self.x.center, self.y.center)

    def get_anchor(self, xanchor, yanchor, direction=None, selbox=None):
        """Calls get_distance with the given anchor options"""
        return self.anchor_distance(getattr(self, XAN[xanchor]), getattr(self, YAN[yanchor]),
                                    direction=direction, selbox=selbox)

    @staticmethod
    def anchor_distance(x, y, direction=0, selbox=None):
        """Using the x,y returns a single sortable value based on direction and angle

        direction - int (custom angle), tb/bt (top/bottom), lr/rl (left/right), ri/ro (radial)
        selbox - The bounding box of the whole selection for radial anchors
        """
        rot = 0
        if isinstance(direction, int):  # Angle
            if direction not in CUSTOM_DIRECTION:
                return hypot(x, y) * (cos(radians(-direction) - atan2(y, x)))
            direction = CUSTOM_DIRECTION[direction]

        if direction in ('ro', 'ri'):
            if selbox is None:
                raise ValueError("Radial distance not available without selection bounding box")
            rot = hypot(selbox.x.center - x, selbox.y.center - y)

        return [y, -y, x, -x, rot, -rot][DIRECTION.index(direction)]


class DirectedLineSegment(object):
    """
    A directed line segment

    DirectedLineSegment(((x0, y0), (x1, y1)))
    """

    start = Vector2d()  # start point of segment
    end = Vector2d()  # end point of segment

    x0 = property(lambda self: self.start.x)  # pylint: disable=invalid-name
    y0 = property(lambda self: self.start.y)  # pylint: disable=invalid-name
    x1 = property(lambda self: self.end.x)
    y1 = property(lambda self: self.end.y)
    dx = property(lambda self: self.x1 - self.x0)  # pylint: disable=invalid-name
    dy = property(lambda self: self.y1 - self.y0)  # pylint: disable=invalid-name

    @overload
    def __init__(self):  # type: () -> None
        pass

    @overload
    def __init__(self, other):  # type: (DirectedLineSegment) -> None
        pass

    @overload
    def __init__(self, start, end):  # type: (VectorLike, VectorLike) -> None
        pass

    def __init__(self, *args):
        if not args:  # overload 0
            start, end = Vector2d(), Vector2d()
        elif len(args) == 1:  # overload 1
            other, = args
            start, end = other.start, other.end
        elif len(args) == 2:  # overload 2
            start, end = args
        else:
            raise ValueError("DirectedLineSegment() can't be constructed from {}".format(args))

        self.start = Vector2d(start)
        self.end = Vector2d(end)

    def __eq__(self, other):
        if isinstance(other, (tuple, DirectedLineSegment)):
            return tuple(self) == tuple(other)
        return False

    def __iter__(self):
        yield self.x0
        yield self.x1
        yield self.y0
        yield self.y1

    @property
    def length(self):
        """Get the length from the top left to the bottom right of the line"""
        return sqrt((self.dx ** 2) + (self.dy ** 2))

    @property
    def angle(self):
        """Get the angle of the line created by this segment"""
        return pi * (atan2(self.dy, self.dx)) / 180

    def distance_to_point(self, x, y):
        """Get the distance to the given point (x, y)"""
        segment2 = DirectedLineSegment(self.start, (x, y))
        dot2 = segment2.dot(self)
        if dot2 <= 0:
            return DirectedLineSegment((x, y), self.start).length
        if self.dot(self) <= dot2:
            return DirectedLineSegment((x, y), self.end).length
        return self.perp_distance(x, y)

    def perp_distance(self, x, y):
        """Perpendicular distance to the given point"""
        if self.length == 0:
            return None
        return fabs((self.dx * (self.y0 - y)) - ((self.x0 - x) * self.dy)) / self.length

    def dot(self, other):  # type: (DirectedLineSegment) -> float
        """Get the dot product with the segment with another"""
        return self.dx * other.dx + self.dy * other.dy

    def point_at_ratio(self, ratio):
        """Get the point at the given ratio along the line"""
        return self.x0 + ratio * self.dx, self.y0 + ratio * self.dy

    def point_at_length(self, length):
        """Get the point as the length along the line"""
        return self.point_at_ratio(length / self.length)

    def parallel(self, x, y):
        """Create parallel Segment"""
        return DirectedLineSegment((x + self.dx, y + self.dy), (x, y))

    def intersect(self, other):  # type: (DirectedLineSegment) -> Optional[Vector2d]
        """Get the intersection between two segments"""
        other = DirectedLineSegment(other)
        denom = (other.dy * self.dx) - (other.dx * self.dy)
        num = (other.dx * (self.y0 - other.y0)) - (other.dy * (self.x0 - other.x0))
        # num2 = (self.width * (self.top - other.top)) - (self.height * (self.left - other.left))

        if denom != 0:
            return Vector2d(
                self.x0 + ((num / denom) * self.dx),
                self.y0 + ((num / denom) * self.dy)
            )
        return None

    def __repr__(self):
        return "DirectedLineSegment(({0.start}), ({0.end}))".format(self)


def cubic_extrema(py0, py1, py2, py3):
    """Returns the extreme value, given a set of bezier coordinates"""

    atol = 1e-9
    cmin, cmax = min(py0, py3), max(py0, py3)
    pd1 = py1 - py0
    pd2 = py2 - py1
    pd3 = py3 - py2

    def _is_bigger(point):
        if (point > 0) and (point < 1):
            pyx = py0 * (1 - point) * (1 - point) * (1 - point) + \
                  3 * py1 * point * (1 - point) * (1 - point) + \
                  3 * py2 * point * point * (1 - point) + \
                  py3 * point * point * point
            return min(cmin, pyx), max(cmax, pyx)
        return cmin, cmax

    if fabs(pd1 - 2 * pd2 + pd3) > atol:
        if pd2 * pd2 > pd1 * pd3:
            pds = sqrt(pd2 * pd2 - pd1 * pd3)
            cmin, cmax = _is_bigger((pd1 - pd2 + pds) / (pd1 - 2 * pd2 + pd3))
            cmin, cmax = _is_bigger((pd1 - pd2 - pds) / (pd1 - 2 * pd2 + pd3))

    elif fabs(pd2 - pd1) > atol:
        cmin, cmax = _is_bigger(-pd1 / (2 * (pd2 - pd1)))

    return cmin, cmax


def quadratic_extrema(py0, py1, py2):
    atol = 1e-9
    cmin, cmax = min(py0, py2), max(py0, py2)

    def _is_bigger(point):
        if (point > 0) and (point < 1):
            pyx = py0 * (1 - point) * (1 - point) + \
                  2 * py1 * point * (1 - point) + \
                  py2 * point * point
            return min(cmin, pyx), max(cmax, pyx)
        return cmin, cmax

    if fabs(py0 + py2 - 2 * py1) > atol:
        cmin, cmax = _is_bigger((py0 - py1) / (py0 + py2 - 2 * py1))

    return cmin, cmax
