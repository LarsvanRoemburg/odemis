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
    import fastem_calibrations
    from fastem_calibrations import autofocus_multiprobe, scan_rotation_pre_align, scan_amplitude_pre_align, \
        image_translation_pre_align, descan_gain, image_rotation_pre_align, image_rotation, image_translation
except ImportError:
    fastem_calibrations = None

# The executor is a single object, independent of how many times the module (fastem.py) is loaded.
_executor = model.CancellableThreadPoolExecutor(max_workers=1)


# TODO pass a list of calibrations to make the code reusable in multiple places?
def align(scanner, multibeam, descanner, detector, ccd, stage, det_rotator):
    """
    :param main_data: (odemis.gui.model.FastEMMainGUIData) TODO
    :returns: (ProgressiveFuture) Alignment future object, which can be cancelled. The result of the future is TODO
    """

    if fastem_calibrations is None:
        raise ModuleNotFoundError("fastem_calibration module missing. Cannot run calibrations.")
    # If raise, then acquisition is still in progress; cancel? If return, fails in caller as no future returned.

    # TODO how to handle the time estimate?
    est_start = time.time() + 0.1
    f = model.ProgressiveFuture(start=est_start,
                                end=est_start + 20)  # Rough time estimation

    # Create a task that runs the calibration and alignments.
    task = CalibrationTask(scanner, multibeam, descanner, detector, ccd, stage, det_rotator, f)

    f.task_canceller = task.cancel  # lets the future know how to cancel the task.

    # Connect the future to the task and run it in a thread.
    # task.run is executed by the executor and runs as soon as no other task is executed
    _executor.submitf(f, task.run)

    return f


class CalibrationTask(object):
    """
    The calibration task.
    """

    def __init__(self, scanner, multibeam, descanner, detector, ccd, stage, det_rotator, future):
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
        self._det_rotator = det_rotator
        self._future = future

        # List of calibrations to be executed. Used for progress update.
        self._calibrations_remaining = {"autofocus_multiprobe", "scan_rotation_pre_align", "scan_amplitude_pre_align",
                                        "descan_gain_static", "image_rotation_pre_align", "image_translation_pre_align",
                                        "image_rotation", "image_translation"}

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
        total_calibration_time = 60  # self.estimate_calibration_time() TODO

        # No need to set the start time of the future: it's automatically done when setting its state to running.
        self._future.set_progress(end=time.time() + total_calibration_time)  # provide end time to future
        logging.info("Starting calibrations, with expected duration of %f s", total_calibration_time)

        dataflow = self._detector.data

        try:
            logging.debug("Starting calibrations.")
            # Note: Run calibrations on bare scintillator
            autofocus_multiprobe.run_autofocus(self._scanner, self._multibeam, self._descanner, self._detector,
                                               dataflow, self._ccd, self._stage)
            self._calibrations_remaining.discard("autofocus-multiprobe")

            # In case the calibrations was cancelled by a client, before the future returned, raise cancellation error.
            if self._cancelled:
                raise CancelledError()

            # Update the time left for the calibrations.
            expected_time = len(self._calibrations_remaining) * 10
            self._future.set_progress(end=time.time() + expected_time)

            # TODO should we call all the calibrations in a separate function?
            # scan_rotation_pre_align.run_scan_rotation_pre_align(self._scanner, self._multibeam, self._descanner,
            #                                                     self._detector, dataflow, self._ccd)
            # self._calibrations_remaining.discard(" scan_rotation_pre_align")
            #
            # scan_amplitude_pre_align.run_scan_amplitude_pre_align(self._scanner, self._multibeam, self._descanner,
            #                                                       self._detector, dataflow, self._ccd)
            # self._calibrations_remaining.discard("scan_amplitude_pre_align")
            #
            # descan_gain.run_descan_gain_static(self._scanner, self._multibeam, self._descanner,
            #                                    self._detector, dataflow, self._ccd)
            # self._calibrations_remaining.discard("descan_gain_static")
            #
            # image_rotation_pre_align.run_image_rotation_pre_align(self._scanner, self._multibeam, self._descanner,
            #                                                       self._detector, dataflow, self._ccd,
            #                                                       self._det_rotator)
            # self._calibrations_remaining.discard("image_rotation_pre_align")
            #
            # image_translation_pre_align.run_image_translation_pre_align(self._scanner, self._multibeam, self._descanner,
            #                                                       self._detector, dataflow, self._ccd)
            # self._calibrations_remaining.discard("image_translation_pre_align")
            #
            # image_rotation.run_image_rotation(self._scanner, self._multibeam, self._descanner,
            #                                   self._detector, dataflow, self._det_rotator)
            # self._calibrations_remaining.discard("image_rotation")
            #
            # image_translation.run_image_translation(self._scanner, self._multibeam, self._descanner,
            #                                         self._detector, dataflow, self._ccd)
            # self._calibrations_remaining.discard("image_translation")

        except CancelledError:  # raised in TODO()
            logging.debug("Calibration was cancelled.")
            raise

        except Exception as ex:
            exception = ex  # let the caller handle the exception
        finally:
            # Remove references to the calibrations once all calibrations are finished/cancelled.
            self._calibrations_remaining.clear()
            logging.debug("Finish calibrations.")

        return exception

    def cancel(self, future):
        """
        Cancels the calibrations.
        :param future: (future) The calibration future.
        :return: (bool) True if cancelled, False if too late to cancel as future is already finished.
        """
        self._cancelled = True

        # TODO how do I stop a calibration while running?
        #  Do we want to allow the user to do so -> yes I guess
        # Report if it's too late for cancellation (and the f.result() will return)
        # if not self._calibrations_remaining:  # TODO
        #     return False

        return True

    # def estimate_calibration_time(self):
    #     """TODO"""
    #     return 10
