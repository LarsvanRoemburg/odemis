# -*- coding: utf-8 -*-
'''
Created on 9 Sep 2015

@author: Kimon Tsitsikas

Copyright © 2015 Kimon Tsitsikas, Delmic

This file is part of Odemis.

Odemis is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License version 2 as published by the Free Software Foundation.

Odemis is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with Odemis. If not, see http://www.gnu.org/licenses/.
'''
from __future__ import division

import glob
import logging
from odemis.driver import powerctrl, semcomedi
import os
import unittest


logger = logging.getLogger().setLevel(logging.DEBUG)

# Export TEST_NOHW=1 to force using only the simulator and skipping test cases
# needing real hardware
TEST_NOHW = (os.environ.get("TEST_NOHW", 0) != 0)  # Default to Hw testing

CLASS = powerctrl.PowerControlUnit
if TEST_NOHW:
    # Test using the simulator
    KWARGS = dict(name="test", role="power_control", powered=["sem", "sed"], pin_map={
                  "sem": 0, "sed": 1}, port="/dev/fake")
else:
    # Test using the hardware
    KWARGS = dict(name="test", role="power_control", powered=["sem", "sed"], pin_map={
                  "sem": 0, "sed": 1}, port="/dev/ttyPMT*")

# Control unit used for PCU testing
CLASS_PCU = CLASS
KWARGS_PCU = KWARGS


# @unittest.skip("faster")
class TestStatic(unittest.TestCase):
    """
    Tests which don't need a component ready
    """
    def test_scan(self):
        # Only test for actual device
        if KWARGS["port"] == "/dev/ttyPMT*":
            devices = CLASS_PCU.scan()
            self.assertGreater(len(devices), 0)

    def test_creation(self):
        """
        Doesn't even try to do anything, just create and delete components
        """
        dev = CLASS_PCU(**KWARGS_PCU)

        self.assertTrue(dev.selfTest(), "self test failed.")
        dev.terminate()

    def test_wrong_device(self):
        """
        Check it correctly fails if the port given is not a PCU.
        """
        # Look for a device with a serial number not starting with 37
        paths = glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*")
        realpaths = set(os.path.join(os.path.dirname(p), os.readlink(p)) for p in glob.glob("/dev/ttyPMT*"))
        for p in paths:
            if p in realpaths:
                continue  # don't try a device which is probably a good one

            kwargsw = dict(KWARGS_PCU)
            kwargsw["port"] = p
            with self.assertRaises(IOError):
                dev = CLASS_PCU(**kwargsw)

# arguments used for the creation of basic components
PCU = CLASS_PCU(**KWARGS_PCU)
CONFIG_SED = {"name": "sed", "role": "sed", "power_supplier": PCU,
              "channel": 5, "limits": [-3, 3]}
CONFIG_BSD = {"name": "bsd", "role": "bsd",
              "channel": 6, "limits": [-0.1, 0.2]}
CONFIG_SCANNER = {"name": "scanner", "role": "ebeam", "limits": [[-5, 5], [3, -3]],
                  "channels": [0, 1], "settle_time": 10e-6, "hfw_nomag": 10e-3,
                  "park": [8, 8]}
CONFIG_SEM2 = {"name": "sem", "role": "sem", "power_supplier": PCU, "device": "/dev/comedi0",
               "children": {"detector0": CONFIG_SED, "detector1": CONFIG_BSD, "scanner": CONFIG_SCANNER}
               }


# @unittest.skip("faster")
class TestPowerControl(unittest.TestCase):
    """
    Tests which need a component ready
    """

    @classmethod
    def setUpClass(cls):
        cls.pcu = PCU
        cls.sem = semcomedi.SEMComedi(**CONFIG_SEM2)

        for child in cls.sem.children.value:
            if child.name == CONFIG_SED["name"]:
                cls.sed = child

    @classmethod
    def tearDownClass(cls):
        cls.pcu.terminate()
        cls.sem.terminate()

    def test_send_cmd(self):
        # Send proper command
        ans = self.pcu._sendCommand("PWR 1 1")
        self.assertEqual(ans, '')

        # Send wrong command
        with self.assertRaises(IOError):
            self.pcu._sendCommand("PWR??")

        # Set value out of range
        with self.assertRaises(IOError):
            self.pcu._sendCommand("PWR 8 1")

        # Send proper set and get command
        self.pcu._sendCommand("PWR 0 1")
        ans = self.pcu._sendCommand("PWR? 0")
        ans_i = int(ans)
        self.assertAlmostEqual(ans_i, 1)

    # @unittest.skip("faster")
    def test_power_supply_va(self):
        self.sed.powerSupply.value = True
        self.assertEqual(self.pcu.supplied.value,
                         {"sem": self.sem.powerSupply.value, "sed": self.sed.powerSupply.value})
        self.sem.powerSupply.value = True
        self.assertEqual(self.pcu.supplied.value,
                         {"sem": self.sem.powerSupply.value, "sed": self.sed.powerSupply.value})
        self.sed.powerSupply.value = False
        self.assertEqual(self.pcu.supplied.value,
                         {"sem": self.sem.powerSupply.value, "sed": self.sed.powerSupply.value})
        self.sem.powerSupply.value = False
        self.assertEqual(self.pcu.supplied.value,
                         {"sem": self.sem.powerSupply.value, "sed": self.sed.powerSupply.value})

if __name__ == "__main__":
    unittest.main()
