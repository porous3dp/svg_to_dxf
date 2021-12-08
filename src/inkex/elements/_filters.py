# -*- coding: utf-8 -*-
#
# Copyright (c) 2020 Martin Owens <doctormo@gmail.com>
#                    Sergei Izmailov <sergei.a.izmailov@gmail.com>
#                    Thomas Holder <thomas.holder@schrodinger.com>
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
# pylint: disable=arguments-differ
"""
Element interface for patterns, filters, gradients and path effects.
"""

from lxml import etree

from ..utils import addNS
from ..transforms import Transform

from ._base import BaseElement

class Filter(BaseElement):
    """A filter (usually in defs)"""
    tag_name = 'filter'

    def add_primitive(self, fe_type, **args):
        """Create a filter primitive with the given arguments"""
        elem = etree.SubElement(self, addNS(fe_type, 'svg'))
        elem.update(**args)
        return elem

    class Primitive(BaseElement):
        pass

    class Blend(Primitive):
        tag_name = 'feBlend'

    class ColorMatrix(Primitive):
        tag_name = 'feColorMatrix'

    class ComponentTransfer(Primitive):
        tag_name = 'feComponentTransfer'

    class Composite(Primitive):
        tag_name = 'feComposite'

    class ConvolveMatrix(Primitive):
        tag_name = 'feConvolveMatrix'

    class DiffuseLighting(Primitive):
        tag_name = 'feDiffuseLighting'

    class DisplacementMap(Primitive):
        tag_name = 'feDisplacementMap'

    class Flood(Primitive):
        tag_name = 'feFlood'

    class GaussianBlur(Primitive):
        tag_name = 'feGaussianBlur'

    class Image(Primitive):
        tag_name = 'feImage'

    class Merge(Primitive):
        tag_name = 'feMerge'

    class Morphology(Primitive):
        tag_name = 'feMorphology'

    class Offset(Primitive):
        tag_name = 'feOffset'

    class SpecularLighting(Primitive):
        tag_name = 'feSpecularLighting'

    class Tile(Primitive):
        tag_name = 'feTile'

    class Turbulence(Primitive):
        tag_name = 'feTurbulence'


class Pattern(BaseElement):
    """Pattern element which is used in the def to control repeating fills"""
    tag_name = 'pattern'
    WRAPPED_ATTRS = BaseElement.WRAPPED_ATTRS + (('patternTransform', Transform),)


class Gradient(BaseElement):
    """A gradient instruction usually in the defs"""
    WRAPPED_ATTRS = BaseElement.WRAPPED_ATTRS + (('gradientTransform', Transform),)


class LinearGradient(Gradient):
    tag_name = 'linearGradient'


class RadialGradient(Gradient):
    tag_name = 'radialGradient'


class PathEffect(BaseElement):
    """Inkscape LPE element"""
    tag_name = 'inkscape:path-effect'
