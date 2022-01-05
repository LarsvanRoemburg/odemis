#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 20 Jul 2021

Copyright © 2021 Philip Winkler, Delmic

This file is part of Odemis.

Odemis is free software: you can redistribute it and/or modify it under the terms
of the GNU General Public License version 2 as published by the Free Software
Foundation.

Odemis is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
Odemis. If not, see http://www.gnu.org/licenses/.
"""
import logging
import threading
import time
from concurrent.futures import CancelledError

from odemis import model

try:
    from fastem_calibrations import autofocus_multiprobe
except ImportError:
    fastem_calibrations = None

# The executor is a single object, independent of how many times the module (fastem.py) is loaded.
_executor = model.CancellableThreadPoolExecutor(max_workers=1)


def align(scanner, multibeam, descanner, detector, ccd, stage):
    """
    :param main_data: (odemis.gui.model.FastEMMainGUIData) TODO
    :returns: (ProgressiveFuture) Alignment future object, which can be cancelled. The result of the future is TODO
    """

    if fastem_calibrations is None:
        raise ModuleNotFoundError("fastem_calibration module missing. Cannot run calibrations.")
    # If raise, then acquisition is still in progress; cancel? If return, fails in caller as no future returned.

    f = model.ProgressiveFuture()

    # Create a task that runs the calibration and alignments.
    task = CalibrationTask(scanner, multibeam, descanner, detector, ccd, stage, f)

    f.task_canceller = task.cancel  # lets the future know how to cancel the task.

    # Connect the future to the task and run it in a thread.
    # task.run is executed by the executor and runs as soon as no other task is executed
    _executor.submitf(f, task.run)

    return f


class CalibrationTask(object):
    """
    The calibration task.
    """

    def __init__(self, scanner, multibeam, descanner, detector, ccd, stage, future):
        """
        :param scanner: (xt_client.Scanner) Scanner component connecting to the XT adapter.
        :param multibeam: (technolution.EBeamScanner) The multibeam scanner component of the acquisition server module.
        :param descanner: (technolution.MirrorDescanner) The mirror descanner component of the acquisition server module.
        :param detector: (technolution.MPPC) The detector object to be used for collecting the image data.
        :param future: (ProgressiveFuture) Acquisition future object, which can be cancelled.
                            (Exception or None): Exception raised during the acquisition or None.
        """
        self._scanner = scanner
        self._multibeam = multibeam
        self._descanner = descanner
        self._detector = detector
        self._ccd = ccd
        self._stage = stage
        self._future = future

        # list of calibrations that still need to be done.
        # self._calibrations_remaining = set(TODO)  # Used for progress update.

        # keep track if future was cancelled or not
        self._cancelled = False

        # Threading event, which keeps track of when image data has been received from the detector.
        # self._data_received = threading.Event()

    def run(self):
        """
        Runs a set of calibration procedures.
        :returns: TODO
            megafield: (list of DataArrays) A list of the raw image data. Each data array (entire field, thumbnail,
                or zero array) represents one single field image within the roa (megafield).
            exception: (Exception or None) Exception raised during the acquisition. If some single field image data has
                already been acquired, exceptions are not raised, but returned.
        :raise:
            Exception: TODO If it failed before any single field images were acquired or if acquisition was cancelled.
        """

        exception = None

        # Get the estimated time for the calibrations.
        total_calibration_time = self.estimate_calibration_time()

        # No need to set the start time of the future: it's automatically done when setting its state to running.
        self._future.set_progress(end=time.time() + total_calibration_time)  # provide end time to future
        logging.info("Starting calibrations, with expected duration of %f s", total_calibration_time)

        dataflow = self._detector.data

        try:
            logging.debug("Starting calibrations.")
            autofocus_multiprobe.run_autofocus(self._ccd, self._stage)

        except CancelledError:  # raised in acquire_roa()
            logging.debug("Acquisition was cancelled.")
            raise

        except Exception as ex:
                raise ex
        finally:
            # Remove references to the calibrations once all calibrations are finished/cancelled.
            # self._calibrations_remaining.clear()
            logging.debug("Finish calibrations.")

        return exception

    def cancel(self, future):
        """
        Cancels the calibrations.
        :param future: (future) The calibration future.
        :return: (bool) True if cancelled, False if too late to cancel as future is already finished.
        """
        self._cancelled = True

        # Report if it's too late for cancellation (and the f.result() will return)
        # if not self._calibrations_remaining:  # TODO
        #     return False

        return True

    def estimate_calibration_time(self):
        """TODO"""
        time.sleep(3)
        return 30*60  # TODO