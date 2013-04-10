#-*- coding: utf-8 -*-
"""
@author: Rinze de Laat

Copyright © 2013 Rinze de Laat, Delmic

This file is part of Odemis.

Odemis is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation, either version 2 of the License, or (at your option) any later
version.

Odemis is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
Odemis. If not, see http://www.gnu.org/licenses/.


### Purpose ###

Provides the, partial dynamically generated, configuration for the settings
panel

"""
import logging
import odemis.gui

from odemis.model import  NotApplicableError


# Default settings for the different components.
# Values in the settings dictionary will be used to steer the default
# behaviours in representing values and the way in which they can be altered.
# All values are optional
# Format:
#   role of component
#       vigilant attribute name
#           label
#              control_type (CONTROL_NONE to hide it)
#              range
#              choices
#              scale
#              type
#              format


def _resolution_from_range(va, conf):
    """ Try and get the maximum value of range and use
    that to construct a list of resolutions
    """
    try:
        logging.debug("Generating resolutions...")
        choices = set([va.value])
        res = va.range[1] # start with max resolution

        for dummy in range(3):
            choices.add(res)
            res = (res[0] // 2, res[1] // 2)

        return sorted(choices) # return a list, to be sure it's in order
    except NotApplicableError:
        return [va.value]

def _binning_1d_from_2d(va, conf):
    """
    Find simple binnings available in one dimension (pixel always square)
    binning provided by a camera is normally a 2-tuple of int
    """
    try:
        choices = set([va.value[0]])
        minbin = max(va.range[0])
        maxbin = min(va.range[1])

        # remove choices not available
        for b in [1, 2, 4]: # all we want at best
            if minbin <= b and b <= maxbin:
                choices.add(b)

        return sorted(choices) # return a list, to be sure it's in order
    except NotApplicableError:
        return [va.value[0]]

# TODO: special settings for the acquisition window? (higher ranges)

CONFIG = {
            "ccd":
            {
                "exposureTime":
                {
                    "control_type": odemis.gui.CONTROL_SLIDER,
                    "scale": "log",
                    "range": (0.01, 3.00),
                    "type": "float",
                },
                "binning":
                {
                    "control_type": odemis.gui.CONTROL_RADIO,
                    "choices": _binning_1d_from_2d,
                    "type": "1d_binning", # means will make sure both dimensions are treated as one
                },
                "resolution":
                {
                    "control_type": odemis.gui.CONTROL_COMBO,
                    "choices": _resolution_from_range,
                },
                "readoutRate":
                {
                    "control_type": odemis.gui.CONTROL_INT,
                },

                # what we don't want to display:
                "targetTemperature":
                {
                    "control_type": odemis.gui.CONTROL_NONE,
                },
                "fanSpeed":
                {
                    "control_type": odemis.gui.CONTROL_NONE,
                },
                "pixelSize":
                {
                    "control_type": odemis.gui.CONTROL_NONE,
                },
            },
            "e-beam":
            {
                "energy":
                {
                    "format": True
                },
                "spotSize":
                {
                    "format": True
                },
                "dwellTime":
                {
                    "control_type": odemis.gui.CONTROL_SLIDER,
                    "range": (1e-9, 0.1),
                    "scale": "log",
                    "type": "float",
                },
                "resolution":
                {
                    "control_type": odemis.gui.CONTROL_COMBO,
                    "choices": _resolution_from_range,
                },
                "magnification": # force using just a text field => it's for copy-paste
                {
                    "control_type": odemis.gui.CONTROL_FLT,
                },
                "pixelSize":
                {
                    "control_type": odemis.gui.CONTROL_TEXT,
                },
            }
        }
