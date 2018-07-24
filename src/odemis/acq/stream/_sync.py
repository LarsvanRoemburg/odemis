# -*- coding: utf-8 -*-
'''
Created on 25 Jun 2014

@author: Éric Piel

Copyright © 2014-2018 Éric Piel, Delmic

This file is part of Odemis.

Odemis is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License version 2 as published by the Free Software Foundation.

Odemis is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with Odemis. If not, see http://www.gnu.org/licenses/.
'''

# This contains "synchronised streams", which handle acquisition from multiple
# detector simultaneously.
# On the SPARC, this allows to acquire the secondary electrons and an optical
# detector simultaneously. In theory, it could support even 3 or 4 detectors
# at the same time, but this is not current supported.
# On the SECOM with a confocal optical microscope which has multiple detectors,
# all the detectors can run simultaneously (each receiving a different wavelength
# band).

from __future__ import division

from abc import ABCMeta, abstractmethod
from concurrent import futures
from concurrent.futures._base import RUNNING, FINISHED, CANCELLED, TimeoutError, \
    CancelledError
from functools import partial
import logging
import math
import numpy
from odemis import model, util
from odemis.acq import leech
from odemis.acq.leech import AnchorDriftCorrector
from odemis.acq.stream._live import LiveStream
import random
import Queue
from odemis.model import MD_POS, MD_DESCRIPTION, MD_PIXEL_SIZE, MD_ACQ_DATE, MD_AD_LIST, \
    MD_DWELL_TIME, MD_POL_MODE

from odemis.util import img, units, spot, executeAsyncTask
import threading
import time

from odemis.model import hasVA
from ._base import Stream, POL_POSITIONS

# On the SPARC, it's possible that both the AR and Spectrum are acquired in the
# same acquisition, but it doesn't make much sense to acquire them
# simultaneously because the two optical detectors need the same light, and a
# mirror is used to select which path is taken. In addition, the AR stream will
# typically have a lower repetition (even if it has same ROI). So it's easier
# and faster to acquire them sequentially.
# TODO: for now, when drift correction is used, it's reset between each MDStream
# acquisition. The same correction should be used for the entire acquisition.
# They all should rely on the same initial anchor acquisition, and keep the
# drift information between them. Possibly, this could be done by passing a
# common DriftEstimator to each MDStream, maybe as a Leech.
# List of detector roles which when acquiring will control the e-beam scanner.
# Note: this is true mostly because we use always the same hardware (ie, DAQ board)
# to get the output synchronised with the e-beam. In theory, this could be
# different for each hardware configuration.
# TODO: Have a way in the microscope model to indicate a detector is synchronised
# with a scanner/emitter.
EBEAM_DETECTORS = ("se-detector", "bs-detector", "cl-detector", "monochromator",
                   "ebic-detector")


class MultipleDetectorStream(Stream):
    """
    Abstract class for all specialised streams which are actually a combination
    of multiple streams acquired simultaneously. The main difference from a
    normal stream is the init arguments are Streams, and .raw is composed of all
    the .raw from the sub-streams.
    """
    __metaclass__ = ABCMeta

    def __init__(self, name, streams):
        """
        streams (list of Streams): they should all have the same emitter (which
          should be the e-beam, and will be used to scan). Streams should have
          a different detector. The order matters.
          The first stream with .repetition will be used to define
          the region of acquisition (ROA), with the .roi and .fuzzing VAs.
          The first stream with .useScanStage will be used to define a scanning
          stage.
          The first leech of type AnchorDriftCorrector will be used for drift correction.
        """
        # TODO: in order to relax the need to have a e-beam related detector,
        # the e-beam scanner should have a way to force the scanning without
        # any acquisition. Maybe by providing a Dataflow which returns no
        # data but supports all the subscribe/synchronisation mechanisms.
        self.name = model.StringVA(name)
        assert len(streams) >= 1
        self._streams = tuple(streams)
        s0 = streams[0]
        self._s0 = s0
        self._det0 = s0._detector
        self._df0 = s0._dataflow

        # Don't use the .raw of the substreams, because that is used for live view
        self._raw = []
        self._anchor_raw = []  # data of the anchor region

        # Emitter should be the same for all the streams
        self._emitter = s0._emitter
        for s in streams[1:]:
            if self._emitter != s.emitter:
                raise ValueError("Streams %s and %s have different emitters" % (s0, s))

        # Get ROA from the first stream with this info
        for s in streams:
            if model.hasVA(s, "repetition") and model.hasVA(s, "roi"):
                logging.debug("Using ROA from %s", s)
                self.repetition = s.repetition
                self.roi = s.roi
                if model.hasVA(s, "fuzzing"):
                    self.fuzzing = s.fuzzing
                break

        # Get optical path manager if found on any of the substreams
        self._opm = None
        for s in self._streams:
            if hasattr(s, "_opm") and s._opm:
                if self._opm and self._opm != s._opm:
                    logging.warning("Multiple different optical path managers were found.")
                    break
                self._opm = s._opm

        # Pick the right scanning stage settings
        for s in streams:
            if model.hasVA(s, "useScanStage") and s._sstage:
                logging.debug("Using scanning stage from %s", s)
                self.useScanStage = s.useScanStage
                self._sstage = s._sstage
                break

        # Get polarization analyzer if found in optical substream
        self._analyzer = None
        for s in streams:
            if hasattr(s, "analyzer") and s.analyzer:
                if self._analyzer:
                    raise ValueError("Only one stream can have an analyzer specified")
                # get polarization analyzer and the VA with the requested position(s)
                self._analyzer = s.analyzer
                self._polarization = s.polarization
                self._acquireAllPol = s.acquireAllPol

        # acquisition end event
        self._acq_done = threading.Event()

        # For the acquisition
        self._acq_lock = threading.Lock()
        self._acq_state = RUNNING
        self._acq_complete = tuple(threading.Event() for s in streams)
        self._acq_thread = None  # thread
        self._acq_rep_tot = 0  # number of acquisitions to do
        self._acq_rep_n = 0  # number of acquisitions so far
        self._prog_sum = 0  # s, for progress time estimation

        # all the data received, in order, for each stream
        self._acq_data = [[] for _ in streams]

        self._acq_min_date = None  # minimum acquisition time for the data to be acceptable

        # Special subscriber function for each stream dataflow
        self._subscribers = []  # to keep a ref
        for i, s in enumerate(self._streams):
            self._subscribers.append(partial(self._onData, i))

        # For the drift correction
        self._dc_estimator = None
        self._current_future = None

        self.should_update = model.BooleanVA(False)
        self.is_active = model.BooleanVA(False)

#     def __del__(self):
#         logging.debug("MDStream %s unreferenced", self.name.value)

    @property
    def streams(self):
        return self._streams

    @property
    def raw(self):
        """
        The raw data of all the streams and the drift correction, in the same
        order as the streams (but not all stream may have generated the same
          number of DataArray).
        """
        # build the .raw from all the substreams
        r = []
        for sr in (self._raw, self._anchor_raw):
            for da in sr:
                if da.shape != (0,):  # don't add empty array
                    r.append(da)
        return r

    @property
    def leeches(self):
        """
        return (tuple of Leech): leeches to be used during acquisition
        """
        # TODO: make it a set, so if streams have the same leech, it's not duplicated
        r = []
        for s in self.streams:
            r.extend(s.leeches)
        return tuple(r)

    @abstractmethod
    def _estimateRawAcquisitionTime(self):
        """
        return (float): time in s for acquiring the whole image, without drift
         correction
        """
        return 0

    def estimateAcquisitionTime(self):
        # Time required without drift correction
        total_time = self._estimateRawAcquisitionTime()

        rep = self.repetition.value
        npixels = numpy.prod(rep)
        dt = total_time / npixels

        # Estimate time spent for the leeches
        for l in self.leeches:
            l_time = l.estimateAcquisitionTime(dt, (rep[1], rep[0]))
            total_time += l_time
            logging.debug("Estimated overhead time for leech %s: %g s / %g s",
                          type(l), l_time, total_time)

        if hasattr(self, "useScanStage") and self.useScanStage.value:
            if self._sstage:
                # It's pretty hard to estimate the move time, as the speed is
                # only the maximum speed and what takes actually most of the time
                # is to stop to the next point.
                # TODO: just get the output of _getScanStagePositions() and add
                # up distances?
                repetition = tuple(self.repetition.value)
                roi = self.roi.value
                width = (roi[2] - roi[0], roi[3] - roi[1])

                # Take into account the "border" around each pixel
                pxs = (width[0] / repetition[0], width[1] / repetition[1])
                lim = (roi[0] + pxs[0] / 2, roi[1] + pxs[1] / 2,
                       roi[2] - pxs[0] / 2, roi[3] - pxs[1] / 2)

                shape = self._emitter.shape
                sem_pxs = self._emitter.pixelSize.value
                sem_fov = shape[0] * sem_pxs[0], shape[1] * sem_pxs[1]
                phy_width = sem_fov[0] * (lim[2] - lim[0]), sem_fov[1] * (lim[3] - lim[1])

                # Count 2x as we need to go back and forth
                tot_dist = (phy_width[0] * repetition[1] + phy_width[1]) * 2
                speed = self._sstage.speed.value["x"]  # consider both axes have same speed

                npixels = numpy.prod(repetition)
                # x2 the move time for compensating the accel/decel + 50 ms per pixel overhead
                move_time = 2 * tot_dist / speed + npixels * 50e-3  # s
                logging.debug("Estimated total scan stage travel distance is %s = %s s",
                              units.readable_str(tot_dist, "m"), move_time)
                total_time += move_time
            else:
                logging.warning("Estimated time cannot take into account scan stage, "
                                "as no scan stage was provided.")

        if hasattr(self, "_polarization"):
            # 5 s extra for moving the polarization analyzer
            total_time += 5
            if self._acquireAllPol.value:
                total_time = total_time * 6

        return total_time

    def acquire(self):
        # Make sure every stream is prepared, not really necessary to check _prepared
        f = self.prepare()
        f.result()

        # Order matters: if same local VAs for emitter (e-beam). The ones from
        # the last stream are used.
        for s in self._streams:
            s._linkHwVAs()

        # TODO: if already acquiring, queue the Future for later acquisition
        if self._current_future is not None and not self._current_future.done():
            raise IOError("Cannot do multiple acquisitions simultaneously")

        if not self._acq_done.is_set():
            if self._acq_thread and self._acq_thread.isAlive():
                logging.debug("Waiting for previous acquisition to fully finish")
                self._acq_thread.join(10)
                if self._acq_thread.isAlive():
                    logging.error("Previous acquisition not ending")

        # Check if a DriftCorrector Leech is available
        for l in self.leeches:
            if isinstance(l, AnchorDriftCorrector):
                logging.debug("Will run drift correction, using leech %s", l)
                self._dc_estimator = l
                break
        else:
            self._dc_estimator = None

        est_start = time.time() + 0.1
        f = model.ProgressiveFuture(start=est_start,
                                    end=est_start + self.estimateAcquisitionTime())
        self._current_future = f
        self._acq_state = RUNNING  # TODO: move to per acquisition
        self._prog_sum = 0
        f.task_canceller = self._cancelAcquisition

        # run task in separate thread
        executeAsyncTask(f, self._runAcquisition, args=(f,))
        return f

    def _updateProgress(self, future, dur, current, tot, bonus=0):
        """
        update end time of future by indicating the time for one new pixel
        future (ProgressiveFuture): future to update
        dur (float): time it took to do this acquisition
        current (1<int<=tot): current number of acquisitions done
        tot (0<int): number of acquisitions
        bonus (0<float): additional time needed (eg, for leeches)
        """
        # Trick: we don't count the first frame because it's often
        # much slower and so messes up the estimation
        if current <= 1:
            return

        self._prog_sum += dur
        ratio = (tot - current) / (current - 1)
        left = self._prog_sum * ratio
        time_assemble = 0.001 * tot  # very rough approximation
        # add some overhead for the end of the acquisition
        tot_left = left + time_assemble + bonus + 0.1
        future.set_progress(end=time.time() + tot_left)

    def _cancelAcquisition(self, future):
        with self._acq_lock:
            if self._acq_state == FINISHED:
                return False  # too late
            self._acq_state = CANCELLED

        logging.debug("Cancelling acquisition of components %s and %s",
                      self._emitter.name, self._streams[-1]._detector.name)

        # Do it in any case, to be sure
        for s, sub in zip(self._streams, self._subscribers):
            s._dataflow.unsubscribe(sub)
        self._df0.synchronizedOn(None)

        # set the events, so the acq thread doesn't wait for them
        for i in range(len(self._streams)):
            self._acq_complete[i].set()

        # Wait for the thread to be complete (and hardware state restored)
        self._acq_done.wait(5)
        return True

    @abstractmethod
    def _adjustHardwareSettings(self):
        """
        Read the stream settings and adapt the SEM scanner accordingly.
        return (float): estimated time per pixel.
        """
        pass

    def _getPixelSize(self):
        """
        Computes the pixel size (based on the repetition, roi and FoV of the
          e-beam). The RepetitionStream does provide a .pixelSize VA, which
          should contain the same value, but that VA is for use by the GUI.
        return (float): pixel size in m. Warning: it's just one value, as both
         X and Y dimensions are always the same.
        """
        epxs = self._emitter.pixelSize.value
        rep = self.repetition.value
        roi = self.roi.value
        eshape = self._emitter.shape
        phy_size_x = (roi[2] - roi[0]) * epxs[0] * eshape[0]  # one dim is enough
        phy_size_y = (roi[3] - roi[1]) * epxs[1] * eshape[1]  # one dim is enough
        pxsy = phy_size_y / rep[1]
        logging.debug("pxs guessed = %s, %s", pxsy, phy_size_x / rep[0])
        return phy_size_x / rep[0]

    def _getSpotPositions(self):
        """
        Compute the positions of the e-beam for each point in the ROI
        return (numpy ndarray of floats of shape (Y,X,2)): each value is for a
          given Y,X in the rep grid -> 2 floats corresponding to the
          translation X,Y. Note that the dimension order is different between
          index and content, because X should be scanned first, so it's last
          dimension in the index.
        """
        rep = tuple(self.repetition.value)
        roi = self.roi.value
        width = (roi[2] - roi[0], roi[3] - roi[1])

        # Take into account the "border" around each pixel
        pxs = (width[0] / rep[0], width[1] / rep[1])
        lim = (roi[0] + pxs[0] / 2, roi[1] + pxs[1] / 2,
               roi[2] - pxs[0] / 2, roi[3] - pxs[1] / 2)

        shape = self._emitter.shape
        # convert into SEM translation coordinates: distance in px from center
        # (situated at 0.5, 0.5), can be floats
        lim_main = (shape[0] * (lim[0] - 0.5), shape[1] * (lim[1] - 0.5),
                    shape[0] * (lim[2] - 0.5), shape[1] * (lim[3] - 0.5))
        logging.debug("Generating points in the SEM area %s, from rep %s and roi %s",
                      lim_main, rep, roi)

        pos = numpy.empty((rep[1], rep[0], 2), dtype=numpy.float)
        posy = pos[:, :, 1].swapaxes(0, 1)  # just a view to have Y as last dim
        posy[:, :] = numpy.linspace(lim_main[1], lim_main[3], rep[1])
        # fill the X dimension
        pos[:, :, 0] = numpy.linspace(lim_main[0], lim_main[2], rep[0])
        return pos

    def _getScanStagePositions(self):
        """
        Compute the positions of the scan stage for each point in the ROI
        return (numpy ndarray of floats of shape (X,Y,2)): each value is for a
          given X/Y in the repetition grid -> 2 floats corresponding to the
          absolute position of the X/Y axes of the stage.
        """
        repetition = tuple(self.repetition.value)
        roi = self.roi.value
        width = (roi[2] - roi[0], roi[3] - roi[1])

        # Take into account the "border" around each pixel
        pxs = (width[0] / repetition[0], width[1] / repetition[1])
        lim = (roi[0] + pxs[0] / 2, roi[1] + pxs[1] / 2,
               roi[2] - pxs[0] / 2, roi[3] - pxs[1] / 2)

        shape = self._emitter.shape
        sem_pxs = self._emitter.pixelSize.value
        sem_fov = shape[0] * sem_pxs[0], shape[1] * sem_pxs[1]

        # Convert into physical translation
        sstage = self._sstage
        saxes = sstage.axes
        spos = sstage.position.value
        spos_rng = (saxes["x"].range[0], saxes["y"].range[0],
                    saxes["x"].range[1], saxes["y"].range[1])  # max phy ROI
        sposc = ((spos_rng[0] + spos_rng[2]) / 2,
                 (spos_rng[1] + spos_rng[3]) / 2)
        dist_c = math.hypot(spos["x"] - sposc[0], spos["y"] - sposc[1])
        if dist_c > 10e-6:
            logging.warning("Scan stage is not initially at center %s, but %s", sposc, spos)

        phy_shift = sem_fov[0] * (0.5 - lim[0]), -sem_fov[1] * (0.5 - lim[1])  # Y is opposite dir
        phy_width = sem_fov[0] * (lim[2] - lim[0]), sem_fov[1] * (lim[3] - lim[1])
        spos0 = spos["x"] - phy_shift[0], spos["y"] - phy_shift[1]
        lim_main = (spos0[0], spos0[1],
                    spos0[0] + phy_width[0], spos0[1] - phy_width[1])
        logging.debug("Generating stage points in the area %s, from rep %s and roi %s with FoV %s",
                      lim_main, repetition, roi, sem_fov)

        if not (spos_rng[0] <= lim_main[0] <= lim_main[2] <= spos_rng[2] and
                spos_rng[1] <= lim_main[3] <= lim_main[1] <= spos_rng[3]):  # Y decreases
            raise ValueError("ROI goes outside the scan stage range (%s > %s)" %
                             (lim_main, spos_rng))

        pos = numpy.empty(repetition + (2,), dtype=numpy.float)
        posx = pos[:, :, 0].swapaxes(0, 1)  # just a view to have X as last dim
        posx[:, :] = numpy.linspace(lim_main[0], lim_main[2], repetition[0])
        # fill the X dimension
        pos[:, :, 1] = numpy.linspace(lim_main[1], lim_main[3], repetition[1])
        return pos

    @abstractmethod
    def _runAcquisition(self, future):
        """
        Acquires images from the multiple detectors via software synchronisation.
        Warning: can be quite memory consuming if the grid is big
        returns (list of DataArray): all the data acquired
        raises:
          CancelledError() if cancelled
          Exceptions if error
        """
        pass

    def _onData(self, n, df, data):
        """
        Callback function. Called for each stream n. Called when detector data is received in
        _acquireImage (when calling self._trigger.notify()).
        :param n (0<=int): the detector/stream index
        :param df (DataFlow): detector's dataflow
        :param data (DataArray): image (2D array) received from detector
        """

        logging.debug("Stream %d data received", n)
        s = self._streams[n]
        if self._acq_min_date > data.metadata.get(model.MD_ACQ_DATE, 0):
            # This is a sign that the e-beam might have been at the wrong (old)
            # position while Rep data is acquiring
            logging.warning("Dropping data because it started %g s too early",
                            self._acq_min_date - data.metadata.get(model.MD_ACQ_DATE, 0))
            # TODO: As the detector is synchronised, we need to restart it.
            # Or maybe not, as the typical reason it arrived early is that the
            # detector was already running, in which case they haven't
            # "consumed" the previous trigger yet ??
            # self._trigger.notify()
            return

        # Only store the first data corresponding to the pixel
        # TODO: If we expect N output / pixel, store all the data received, and
        # average it (to reduce noise) or at least in case of fuzzing, store and
        # average the N expected images
        if not self._acq_complete[n].is_set():
            self._acq_data[n].append(data)  # append image acquired from detector to list
            self._acq_complete[n].set()  # indicate the data has been received

    def _preprocessData(self, n, data, i):
        """
        Preprocess the raw data, just after it was received from the detector.
        Note: this version just return the data as is. Override it to do
          something more advanced.
        n (0<=int): the detector/stream index
        data (DataArray): the data as received from the detector, from
          _onData(), and with MD_POS updated to the current position of the e-beam.
        i (int, int): iteration number in X, Y
        return (value): value as needed by _onCompletedData
        """
        return data

    def _onCompletedData(self, n, raw_das):
        """
        Called at the end of an entire acquisition. It should assemble the data
        and append it to ._raw .
        Override if you need to process the data in a different way.
        n (0<=int): the detector/stream index
        raw_das (list of DataArray): data as received from the detector.
           The data is ordered, with X changing fast, then Y slow
        """

        # Default is to assume the data is 2D and assemble it.
        da = self._assemble2DData(self.repetition.value, raw_das)
        # explicitly add names to make sure they are different
        da.metadata[MD_DESCRIPTION] = self._streams[n].name.value
        self._raw.append(da)

    def _get_center_pxs(self, rep, sub_shape, datatl):
        """
        Computes the center and pixel size of the entire data based on the
        top-left data acquired.
        rep (int, int): number of pixels (tiles) in X, Y
        sub_shape (int, int): number of sub-pixels in a pixel
        datatl (DataArray): first data array acquired
        return:
            center (tuple of floats): position in m of the whole data
            pxs (tuple of floats): pixel size in m of the sub-pixels
        """
        # Compute center of area, based on the position of the first point (the
        # position of the other points can be wrong due to drift correction)
        center_tl = datatl.metadata[MD_POS]
        dpxs = datatl.metadata[MD_PIXEL_SIZE]
        tl = (center_tl[0] - (dpxs[0] * (datatl.shape[-1] - 1)) / 2,
              center_tl[1] + (dpxs[1] * (datatl.shape[-2] - 1)) / 2)
        logging.debug("Computed center of top-left pixel at at %s", tl)

        # Note: we don't rely on the MD_PIXEL_SIZE, because if the e-beam was in
        # spot mode (res 1x1), the scale is not always correct, which gives an
        # incorrect metadata.
        pxsx = self._getPixelSize()
        pxs = pxsx / sub_shape[0], pxsx / sub_shape[1]
        trep = rep[0] * sub_shape[0], rep[1] * sub_shape[1]
        center = (tl[0] + (pxs[0] * (trep[0] - 1)) / 2,
                  tl[1] - (pxs[1] * (trep[1] - 1)) / 2)
        logging.debug("Computed data width to be %s x %s, with center at %s",
                      pxs[0] * rep[0], pxs[1] * rep[1], center)

        if numpy.prod(datatl.shape) > 1:
            # pxs and dpxs ought to be identical
            if not util.almost_equal(pxs[0], dpxs[0]):
                logging.warning("Expected pixel size of %s, but data has %s",
                                pxs, dpxs)

        return center, pxs

    def _assemble2DData(self, rep, data_list):
        """
        Take all the data received from a 0D DataFlow and assemble it in a
        2D image. If each acquisition from the DataFlow is more than a point,
        use _assembleTiles().

        rep (tuple of 2 0<ints): X/Y repetition
        data_list (list of M DataArray of any shape): all the data received,
          with X varying first, then Y. Each DataArray may be of different shape.
          If a DataArray is bigger than a single pixel, it is flatten and each
          value is considered consecutive.
          The MD_POS and MD_PIXEL_SIZE of the first DataArray is used to compute
          the metadata of the complete image.
        return (DataArray of shape rep[1], rep[0]): the 2D reconstruction
        """
        assert len(data_list) > 0

        # If the detector generated no data, just return no data
        # This currently happens with the semcomedi counters, which cannot
        # acquire simultaneously analog input.
        if data_list[0].shape == (0,):
            if not all(d.shape == (0,) for d in data_list):
                logging.warning("Detector received mix of empty and non-empty data")
            return data_list[0]

        # start with the metadata from the first point
        md = data_list[0].metadata.copy()
        center, pxs = self._get_center_pxs(rep, (1, 1), data_list[0])
        md.update({MD_POS: center,
                   MD_PIXEL_SIZE: pxs})

        # concatenate data into one big array of (number of pixels,1)
        flat_list = [ar.flatten() for ar in data_list]
        main_data = numpy.concatenate(flat_list)
        logging.debug("Assembling %s points into %s shape", main_data.shape, rep)
        # reshape to (Y, X)
        main_data.shape = rep[::-1]
        main_data = model.DataArray(main_data, metadata=md)
        return main_data

    def _assembleTiles(self, rep, data_list):
        """
        Convert a series of tiles acquisitions into an image (2D)
        rep (2 x 0<ints): Number of tiles in the output (Y, X)
        data_list (list of N DataArray of shape T, S): the values,
            ordered in blocks of TxS with X first, then Y. N = Y*X.
            Each element along N is tiled on the final data.
            If multiple images were recorded per pixel position, the number of pixels X*Y (scan positions)
            does not match len(data_list). Every multiple of X*Y represents the same pixel (scan position).
            Multiple scans per pixel will be averaged.
        return (DataArray of shape Y*T, X*S): the data with the correct metadata
        """
        # N = len(data_list)
        T, S = data_list[0].shape
        X, Y = rep
        # copy into one big array N, Y, X
        arr = numpy.array(data_list)

        if T == 1 and S == 1:
            # fast path: the data is already ordered just copy
            # reshape to get a 2D image
            # check if number of px scans (rep) is equal to number of images acquired (arr)
            # else: multiple images for same pixel were acquired (e.g. multiple polarization settings)
            if numpy.prod(rep) == arr.shape[0]:
                arr.shape = rep[::-1]
            else:
                im_px = int(arr.shape[0]/numpy.prod(rep))  # number of images per pixel
                arr.shape = rep[::-1] + (im_px,)
                # average images
                arr = numpy.mean(arr, 2).astype(data_list[0].dtype)
        else:
            # need to reorder data by tiles
            # change N to Y, X
            arr.shape = (Y, X, T, S)
            # change to Y, T, X, S by moving the "T" axis
            arr = numpy.rollaxis(arr, 2, 1)
            # and apply the change in memory (= 1 copy)
            arr = numpy.ascontiguousarray(arr)
            # reshape to apply the tiles
            arr.shape = (Y * T, X * S)

        # start with the metadata from the first point
        md = data_list[0].metadata.copy()
        center, pxs = self._get_center_pxs(rep, (T, S), data_list[0])
        md.update({MD_POS: center,
                   MD_PIXEL_SIZE: pxs})

        return model.DataArray(arr, md)

    def _assembleAnchorData(self, data_list):
        """
        Take all the data acquired for the anchor region

        data_list (list of N DataArray of shape 2D (Y, X)): all the anchor data
        return (DataArray of shape (1, N, 1, Y, X))
        """
        assert len(data_list) > 0
        assert data_list[0].ndim == 2

        # extend the shape to TZ dimensions to allow the concatenation on T
        for d in data_list:
            d.shape = (1, 1) + d.shape

        anchor_data = numpy.concatenate(data_list)
        anchor_data.shape = (1,) + anchor_data.shape

        # copy the metadata from the first image (which contains the original
        # position of the anchor region, without drift correction)
        md = data_list[0].metadata.copy()
        md[MD_DESCRIPTION] = "Anchor region"
        md[MD_AD_LIST] = tuple(d.metadata[MD_ACQ_DATE] for d in data_list)
        return model.DataArray(anchor_data, metadata=md)

    def _startLeeches(self, px_time, rep, tot_num):
        """
        A leech can be drift correction (dc) and/or probe/sample current (pca).
        During the leech time the drift correction and/or the probe current measurements are conducted.
        Start a counter np (= next pixel) until a leech is called.
        Leech is called after a well specified number of acquired pixel positions.
        :param px_time (0<float): estimated time spend for one pixel
        :param rep (int, int): 2D grid of pixel positions to be acquired
        :param tot_num (int): total number of e-beam positions
        :return:
            leech_np (list of 0<int or None): for each leech, number of pixels before the leech should be
                                              executed again. It's automatically updated inside the list.
                                              (np = next pixels)
            leech_time_ppx (float): extra time needed on average for a single pixel for all leches (s)
        """

        leech_np = []
        leech_time = 0  # how much time leeches will cost
        for l in self.leeches:
            try:
                leech_time += l.estimateAcquisitionTime(px_time, (rep[1], rep[0]))
                np = l.start(px_time, (rep[1], rep[0]))  # np = next pixel = counter until execution
            except Exception:
                logging.exception("Leech %s failed to start, will be disabled for this acquisition", l)
                np = None
            leech_np.append(np)

        # extra time needed on average for a single pixel for all leches (s)
        leech_time_ppx = leech_time / tot_num  # s/px

        return leech_np, leech_time_ppx

    def _stopLeeches(self):
        """
        Stop the leeches after all pixels are acquired.
        """
        for l in self.leeches:
            l.complete(self.raw)


class SEMCCDMDStream(MultipleDetectorStream):
    """
    Abstract class for multiple detector Stream made of SEM + CCD.
    It handles acquisition, but not rendering (so there is no .image).
    The acquisition is software synchronised. The acquisition code takes care of
    moving the SEM spot and starts a new CCD acquisition at each spot. It brings
    a bit more overhead than linking directly the event of the SEM to the CCD
    detector trigger, but it's very reliable.
    """

    def __init__(self, name, streams):
        """
        streams (list of Streams): in addition to the requirements of
          MultipleDetectorStream, there should be precisely two streams. The
          first one MUST be controlling the SEM e-beam, while the last stream
          should be have a Camera as detector (ie, with .exposureTime).
        """

        # TODO: Support multiple SEM streams.
        # Note: supporting multiple cameras in most case is not useful because
        # a) each detector would need to have about the same exposure time/readout
        # b) each detector should receive light for the single optical path
        # (That can happen for instance on confocal microscopes, but anyway we
        # have a special stream for that).

        super(SEMCCDMDStream, self).__init__(name, streams)

        if self._det0.role not in EBEAM_DETECTORS:
            raise ValueError("First stream detector %s doesn't control e-beam" %
                             (self._det0.name,))

        # TODO: For now only support 2 streams, linked to the e-beam and CCD-based
        if len(streams) != 2:
            raise ValueError("Requires exactly 2 streams")

        s1 = streams[1]  # detector stream
        if not model.hasVA(s1._detector, "exposureTime"):
            raise ValueError("%s detector '%s' doesn't seem to be a CCD" %
                             (s1, s1._detector.name,))

        self._sccd = s1
        self._ccd = s1._detector
        self._ccd_df = s1._dataflow
        self._trigger = self._ccd.softwareTrigger
        self._ccd_idx = len(self._streams) - 1  # optical detector is always last in streams

    def _estimateRawAcquisitionTime(self):
        """
        return (float): time in s for acquiring the whole image, without drift
         correction
        """
        try:
            # Each pixel x the exposure time (of the detector) + readout time +
            # 30ms overhead + 20% overhead
            try:
                ro_rate = self._sccd._getDetectorVA("readoutRate").value
            except Exception:
                ro_rate = 100e6  # Hz
            res = self._sccd._getDetectorVA("resolution").value
            readout = numpy.prod(res) / ro_rate

            exp = self._sccd._getDetectorVA("exposureTime").value
            dur_image = (exp + readout + 0.03) * 1.20
            duration = numpy.prod(self.repetition.value) * dur_image
            # Add the setup time
            duration += self.SETUP_OVERHEAD

            return duration
        except Exception:
            msg = "Exception while estimating acquisition time of %s"
            logging.exception(msg, self.name.value)
            return Stream.estimateAcquisitionTime(self)

    def _adjustHardwareSettings(self):
        """
        Read the SEM and CCD stream settings and adapt the SEM scanner
        accordingly.
        return (float): estimated time for a whole CCD image
        """
        exp = self._sccd._getDetectorVA("exposureTime").value  # s
        rep_size = self._sccd._getDetectorVA("resolution").value
        readout = numpy.prod(rep_size) / self._sccd._getDetectorVA("readoutRate").value

        fuzzing = (hasattr(self, "fuzzing") and self.fuzzing.value)
        if fuzzing:
            # Pick scale and dwell-time so that the (big) pixel is scanned twice
            # fully during the exposure. Scanning twice (instead of once) ensures
            # that even if the exposure is slightly shorter than expected, we
            # still get some signal from everywhere. It could also help in case
            # the e-beam takes too much time to settle at the beginning of the
            # scan, so that the second scan compensates a bit (but for now, we
            # discard the second scan data :-( )

            # Largest (square) resolution the dwell time permits
            rng = self._emitter.dwellTime.range
            pxs = self._getPixelSize()
            max_tile_shape_dt = int(math.sqrt(exp / (rng[0] * 2)))
            # Largest resolution the SEM scale permits (assuming min scale = 1)
            sem_pxs = self._emitter.pixelSize.value
            max_tile_shape_scale = int(pxs / sem_pxs[0])

            # the min of both is the real maximum we can do
            ts = max(1, min(max_tile_shape_dt, max_tile_shape_scale))
            tile_shape = (ts, ts)
            subpxs = pxs / ts
            dt = (exp / numpy.prod(tile_shape)) / 2
            scale = (subpxs / sem_pxs[0], subpxs / sem_pxs[1])

            # Double check fuzzing would work (and make sense)
            if ts == 1 or not (rng[0] <= dt <= rng[1]) or scale[0] < 1 or scale[1] < 1:
                logging.info("Disabled fuzzing because SEM wouldn't support it")
                fuzzing = False

        if fuzzing:
            logging.info("Using fuzzing with tile shape = %s", tile_shape)
            # Handle fuzzing by scanning tile instead of spot
            self._emitter.scale.value = scale
            self._emitter.resolution.value = tile_shape  # grid scan
            self._emitter.dwellTime.value = self._emitter.dwellTime.clip(dt)
        else:
            # Set SEM to spot mode, without caring about actual position (set later)
            self._emitter.scale.value = (1, 1)  # min, to avoid limits on translation
            self._emitter.resolution.value = (1, 1)
            # Dwell time as long as possible, but better be slightly shorter than
            # CCD to be sure it is not slowing thing down.
            self._emitter.dwellTime.value = self._emitter.dwellTime.clip(exp + readout)

        return exp + readout

    def _onCompletedData(self, n, raw_das):
        """
        Called at the end of an entire acquisition. It should assemble the data
        and append it to ._raw .
        Override if you need to process the data in a different way.
        n (0<=int): the detector/stream index
        raw_das (list of DataArray): data as received from the detector.
           The data is ordered, with X changing fast, then Y slow
        """

        # Default is to assume the data is 2D and assemble it.
        da = self._assembleTiles(self.repetition.value, raw_das)

        # explicitly add names of acquisition to make sure they are different
        da.metadata[MD_DESCRIPTION] = self._streams[n].name.value

        self._raw.append(da)

    def _runAcquisition(self, future):
        """
        Acquires images from the multiple detectors via software synchronisation.
        Select whether the ebeam is moved for scanning or the sample stage.
        """

        if hasattr(self, "useScanStage") and self.useScanStage.value:
            return self._runAcquisitionScanStage(future)
        else:
            return self._runAcquisitionEbeam(future)

    def _runAcquisitionEbeam(self, future):
        """
        Acquires images from the multiple detectors via software synchronisation.
        Acquires images via moving the ebeam.
        Warning: can be quite memory consuming if the grid is big
        returns (list of DataArray): all the data acquired
        raises:
          CancelledError() if cancelled
          Exceptions if error
        """
        # TODO: handle better very large grid acquisition (than memory oops)
        try:
            self._acq_done.clear()
            px_time = self._adjustHardwareSettings()
            dwell_time = self._emitter.dwellTime.value
            sem_time = dwell_time * numpy.prod(self._emitter.resolution.value)
            spot_pos = self._getSpotPositions()  # list of center positions for each point of the ROI
            logging.debug("Generating %dx%d spots for %g (dt=%g) s",
                          spot_pos.shape[1], spot_pos.shape[0], px_time, dwell_time)
            rep = self.repetition.value  # (int, int): number of pixels in the ROI (X, Y)
            tot_num = numpy.prod(rep)  # total number of pixels
            sub_pxs = self._emitter.pixelSize.value  # sub-pixel size

            self._acq_data = [[] for _ in self._streams]  # just to be sure it's really empty
            self._raw = []
            self._anchor_raw = []
            logging.debug("Starting repetition stream acquisition with components %s",
                          ", ".join(s._detector.name for s in self._streams))

            # initialize leeches
            leech_np, leech_time_ppx = self._startLeeches(px_time, rep, tot_num)

            # The acquisition works the following way:
            # * The CCD is set to synchronised acquisition, and for every e-beam
            #   spot (or set of sub-pixels in fuzzing mode).
            # * The e-beam synchronised detector(s) is configured for one e-beam
            #   spot and stopped (after a couple of scans) as soon as the CCD
            #   data comes in.
            # Rationale: using the .newPosition Event on the e-beam is not
            # reliable enough as the CCD driver may not receive the data in time.
            # (it might be solvable for most hardware by improving the drivers
            # to put the CCD into special "burst" mode). We could almost use
            # .get() on the CCD, but it's slow, and it's not cancellable. If we
            # use synchronisation also on the e-beam, we cannot stop the scan
            # immediately after the CCD image is received. So we would either
            # stop it a little before (and in fuzzing it might not have scanned
            # everything during the exposure, and can only scan once) or wait
            # one scan too long, which would correspond to almost 50% exposure
            # time overhead per pixel.
            # TODO: between each spot, the e-beam will go back to park position,
            # which might cause some wiggling in the next spot (sub-pixels).
            # Ideally, the spot would just wait at the last pixel of the scan
            # (or the first pixel of the next scan). => use data from the two
            # scans (and check if there is time for more scans during the
            # readout). => Force the ebeam to not park (either by temporarily
            # providing another rest position) or by doing synchronised
            # acquisition (either with just 1 scan, or multiple scans +
            # retrigger, or unsynchronise/resynchronise just before the end of
            # last scan).

            # prepare detector
            self._ccd_df.synchronizedOn(self._trigger)
            # subscribe to last entry in _subscribers (optical detector)
            self._ccd_df.subscribe(self._subscribers[self._ccd_idx])

            # Instead of subscribing/unsubscribing to the SEM for each pixel,
            # we've tried to keep subscribed, but request to be unsynchronised/
            # synchronised. However, synchronizing doesn't cancel the current
            # scanning, so it could still be going on with the old translation
            # while starting the next acquisition.

            # if no polarimetry hardware present
            pos_polarizations = [None]

            # check if polarization VA exists, overwrite list of polarization value
            if self._analyzer:
                if self._acquireAllPol.value:
                    pos_polarizations = POL_POSITIONS
                    logging.debug("Will acquire the following polarization positions: %s" % list(pos_polarizations))
                else:
                    pos_polarizations = [self._polarization.value]
                    logging.debug("Will acquire the following polarization position: %s" % pos_polarizations)

            for pol_pos in pos_polarizations:
                if pol_pos is not None:
                    logging.debug("Acquire with the following polarization position: %s" % pol_pos)
                    # move polarization analyzer to position specified
                    f = self._analyzer.moveAbs({"pol": pol_pos})
                    f.result()

                n = 0  # number of points (ebeam positions) acquired so far

                # iterate over pixel positions for scanning
                for px_idx in numpy.ndindex(*rep[::-1]):  # last dim (X) iterates first
                    trans = tuple(spot_pos[px_idx])  # spot position

                    # take care of drift
                    if self._dc_estimator:
                        trans = (trans[0] - self._dc_estimator.tot_drift[0],
                                 trans[1] - self._dc_estimator.tot_drift[1])
                    cptrans = self._emitter.translation.clip(trans)
                    if cptrans != trans:
                        if self._dc_estimator:
                            logging.error("Drift of %s px caused acquisition region out "
                                          "of bounds: needed to scan spot at %s.",
                                          self._dc_estimator.tot_drift, trans)
                        else:
                            logging.error("Unexpected clipping in the scan spot position %s", trans)
                    self._emitter.translation.value = cptrans
                    logging.debug("E-beam spot after drift correction: %s",
                                  self._emitter.translation.value)
                    logging.debug("Scanning resolution is %s and scale %s",
                                  self._emitter.resolution.value,
                                  self._emitter.scale.value)

                    # acquire
                    self._acquireImage(n, pol_pos, px_idx, px_time, sem_time, sub_pxs,
                                       tot_num, leech_time_ppx, leech_np, future)
                    n += 1

            # acquisition done!
            for s, sub in zip(self._streams, self._subscribers):
                s._dataflow.unsubscribe(sub)
            self._ccd_df.synchronizedOn(None)

            with self._acq_lock:
                if self._acq_state == CANCELLED:
                    raise CancelledError()
                self._acq_state = FINISHED

            for n, das in enumerate(self._acq_data):
                self._onCompletedData(n, das)

            self._stopLeeches()

            if self._dc_estimator:
                self._anchor_raw.append(self._assembleAnchorData(self._dc_estimator.raw))

        except Exception as exp:
            if not isinstance(exp, CancelledError):
                logging.exception("Software sync acquisition of multiple detectors failed")

            # make sure it's all stopped
            for s, sub in zip(self._streams, self._subscribers):
                s._dataflow.unsubscribe(sub)
            self._ccd_df.synchronizedOn(None)

            self._raw = []
            self._anchor_raw = []
            if not isinstance(exp, CancelledError) and self._acq_state == CANCELLED:
                logging.warning("Converting exception to cancellation")
                raise CancelledError()
            raise
        else:
            return self.raw
        finally:
            for s in self._streams:
                s._unlinkHwVAs()
            self._acq_data = [[] for _ in self._streams]  # regain a bit of memory
            self._dc_estimator = None
            self._current_future = None
            self._acq_done.set()

    def _waitForImage(self, px_time):
        """
        Wait for the detector to acquire the image
        :param det_idx (int): index of detector-stream in streams
        :param px_time (0<float): estimated time spend for one pixel
        :return (bool): True if acquisition timed out
        """

        # A big timeout in the wait can cause up to 50 ms latency.
        # => after waiting the expected time only do small waits

        start = time.time()
        endt = start + px_time * 3 + 5
        timedout = not self._acq_complete[self._ccd_idx].wait(px_time + 0.01)
        if timedout:
            logging.debug("Waiting a bit more for detector %d to acquire image." % self._ccd_idx)
            while time.time() < endt:
                timedout = not self._acq_complete[self._ccd_idx].wait(0.005)
                if not timedout:
                    break
        logging.debug("Got synchronized acquisition from detector %d." % self._ccd_idx)

        return timedout

    def _acquireImage(self, n, pol_pos, px_idx, px_time, sem_time, sub_pxs,
                      tot_num, leech_time_ppx, leech_np, future):
        """
        acquires image from detector
        :param n (int): number of points (pixel positions) acquired so far
        :param pol_pos (str): polarization position (None if no polarization hardware)
        :param px_idx (int, int): current scanning position of ebeam
        :param px_time (0<float): expected time spend for one pixel
        :param sem_time (0<float): expected time spend for all sub-pixel
               (=px_time if not fuzzing, and < px_time if fuzzing)
        :param sub_pxs (float, float): sub-pixel size when acquiring in fuzzy-mode
        :param tot_num (int): total number of e-beam positions
        :param leech_time_ppx (float): extra time needed on average for a single pixel for all leches (s)
        :param leech_np (list of 0<int or None): for each leech, number of pixels before the leech should be
           executed again. It's automatically updated inside the list. (np = next pixels)
        :param future: current future running for the whole acquisition
        """

        failures = 0  # keeps track of acquisition failures
        while True:  # Done only once normally, excepted in case of failures
            start = time.time()
            self._acq_min_date = start
            for ce in self._acq_complete:
                ce.clear()

            if self._acq_state == CANCELLED:
                raise CancelledError()

            # subscribe to _subscribers
            for s, sub in zip(self._streams[:-1], self._subscribers[:-1]):
                s._dataflow.subscribe(sub)
            # TODO: in theory (aka in a perfect world), the ebeam would immediately
            # be at the requested position after the subscription starts. However,
            # that's not exactly the case due to:
            # * physics limits the speed of voltage change in the ebeam column,
            #   so it takes the "settle time" before the beam is at the right
            #   place (in the order of 10 µs).
            # * the (odemis) driver is asynchronous, and between the moment it
            #   receives the request to start and the actual moment it asks the
            #   hardware to change voltages, several ms might have passed.
            # One thing that would help is to not park the e-beam between each
            # spot. This way, the ebeam would reach the position much quicker,
            # and if it's not yet at the right place, it's still not that far.
            # The driver could also request the ebeam to the first spot position
            # as soon as the subscription is received.
            # In the meantime, waiting a tiny bit ensures the CCD receives the
            # right data.
            time.sleep(10e-3)  # give more chances spot has been already processed

            # send event to detector to acquire one image
            self._trigger.notify()

            # wait for detector to acquire image
            timedout = self._waitForImage(px_time)

            if self._acq_state == CANCELLED:
                raise CancelledError()

            # Check whether it went fine (= not too long and not too short)
            dur = time.time() - start
            if timedout or dur < px_time * 0.95:
                if timedout:
                    # Note: it can happen we don't receive the data if there
                    # no more memory left (without any other warning).
                    # So we log the memory usage here too.
                    # TODO: Support also for Windows
                    import odemis.util.driver as udriver
                    memu = udriver.readMemoryUsage()
                    # Too bad, need to use VmSize to get any good value
                    logging.warning("Acquisition of repetition stream for "
                                    "pixel %s timed out after %g s. "
                                    "Memory usage is %d. Will try again",
                                    px_idx, px_time * 3 + 5, memu)
                else:  # too fast to be possible (< the expected time - 5%)
                    logging.warning("Repetition stream acquisition took less than %g s: %g s, will try again",
                                    px_time, dur)
                failures += 1
                if failures >= 3:
                    # In three failures we just give up
                    raise IOError("Repetition stream acquisition repeatedly fails to synchronize")
                else:
                    for s, sub, ad in zip(self._streams, self._subscribers, self._acq_data):
                        s._dataflow.unsubscribe(sub)
                        # Ensure we don't keep the data for this run
                        ad[:] = ad[:n]

                    # Restart the acquisition, hoping this time we will synchronize
                    # properly
                    time.sleep(1)
                    self._ccd_df.subscribe(self._subscribers[self._ccd_idx])
                    continue

            # Normally, the SEM acquisitions have already completed
            # get image for SEM streams (at least one for ebeam)
            for s, sub, ce in zip(self._streams[:-1], self._subscribers[:-1], self._acq_complete[:-1]):
                if not ce.wait(sem_time * 1.5 + 5):
                    raise TimeoutError("Acquisition of SEM pixel %s timed out after %g s"
                                       % (px_idx, sem_time * 1.5 + 5))
                logging.debug("Got synchronisation from %s", s)
                s._dataflow.unsubscribe(sub)

            if self._acq_state == CANCELLED:
                raise CancelledError()

            # MD_POS default to the center of the stage, but it needs to be
            # the position of the e-beam (without the shift for drift correction)
            raw_pos = self._acq_data[0][-1].metadata[MD_POS]
            drift_shift = self._dc_estimator.tot_drift if self._dc_estimator else (0, 0)
            cor_pos = (raw_pos[0] + drift_shift[0] * sub_pxs[0],
                       raw_pos[1] - drift_shift[1] * sub_pxs[1])  # Y is upside down
            ccd_data = self._acq_data[self._ccd_idx][-1]
            ccd_data.metadata[MD_POS] = cor_pos

            if self._analyzer:
                ccd_data.metadata[MD_POL_MODE] = pol_pos

            self._acq_data[-1][-1] = self._preprocessData(len(self._streams) - 1, ccd_data, px_idx)
            logging.debug("Processed CCD data %d = %s", n, px_idx)

            leech_time_left = (tot_num - n + 1) * leech_time_ppx
            self._updateProgress(future, time.time() - start, n + 1, tot_num, leech_time_left)

            # Check if it's time to run a leech
            for li, l in enumerate(self.leeches):
                if leech_np[li] is None:
                    continue
                leech_np[li] -= 1
                if leech_np[li] == 0:
                    try:
                        np = l.next([d[-1] for d in self._acq_data])
                    except Exception:
                        logging.exception("Leech %s failed, will retry next pixel", l)
                        np = 1  # try again next pixel
                    leech_np[li] = np
                    if self._acq_state == CANCELLED:
                        raise CancelledError()

            # Since we reached this point means everything went fine, so
            # no need to retry
            break

    def _adjustHardwareSettingsScanStage(self):
        """
        Read the SEM and CCD stream settings and adapt the SEM scanner
        accordingly.
        return (float): estimated time for a whole CCD image
        """
        # Move ebeam to the center
        self._emitter.translation.value = (0, 0)

        return self._adjustHardwareSettings()

    def _runAcquisitionScanStage(self, future):
        """
        Acquires images from the multiple detectors via software synchronisation,
        with a scan stage.
        Warning: can be quite memory consuming if the grid is big
        returns (list of DataArray): all the data acquired
        raises:
          CancelledError() if cancelled
          Exceptions if error
        """
        # The idea of the acquiring with a scan stage:
        #  (Note we expect the scan stage to be about at the center of its range)
        #  * Move the ebeam to 0, 0 (center), for the best image quality
        #  * Start CCD acquisition with software synchronisation
        #  * Move to next position with the stage and wait for it
        #  * Start SED acquisition and trigger CCD
        #  * Wait for the CCD/SED data
        #  * Repeat until all the points have been scanned
        #  * Move back the stage to center

        sstage = self._sstage
        try:
            if not sstage:
                raise ValueError("Cannot acquire with scan stage, as no stage was provided")
            saxes = sstage.axes
            orig_spos = sstage.position.value  # TODO: need to protect from the stage being outside of the axes range?
            prev_spos = orig_spos.copy()
            spos_rng = (saxes["x"].range[0], saxes["y"].range[0],
                        saxes["x"].range[1], saxes["y"].range[1])  # max phy ROI

            self._acq_done.clear()
            px_time = self._adjustHardwareSettingsScanStage()
            dwell_time = self._emitter.dwellTime.value
            sem_time = dwell_time * numpy.prod(self._emitter.resolution.value)
            stage_pos = self._getScanStagePositions()
            logging.debug("Generating %s pos for %g (dt=%g) s",
                          stage_pos.shape[:2], px_time, dwell_time)
            rep = self.repetition.value  # (int, int): 2D grid of pixel positions to be acquired
            sub_pxs = self._emitter.pixelSize.value  # sub-pixel size
            self._acq_data = [[] for _ in self._streams]  # just to be sure it's really empty
            self._raw = []
            self._anchor_raw = []
            logging.debug("Starting repetition stream acquisition with components %s and scan stage %s",
                          ", ".join(s._detector.name for s in self._streams), sstage.name)
            logging.debug("Scanning resolution is %s and scale %s",
                          self._emitter.resolution.value,
                          self._emitter.scale.value)

            tot_num = numpy.prod(rep)

            # initialize leeches
            leech_np, leech_time_ppx = self._startLeeches(px_time, rep, tot_num)

            # Synchronise the CCD on a software trigger
            self._ccd_df.synchronizedOn(self._trigger)
            self._ccd_df.subscribe(self._subscribers[self._ccd_idx])

            n = 0  # number of points acquired so far
            for px_idx in numpy.ndindex(*rep[::-1]):  # last dim (X) iterates first
                # Move the scan stage to the next position
                spos = stage_pos[px_idx[::-1]][0], stage_pos[px_idx[::-1]][1]
                # TODO: apply drift correction on the ebeam. As it's normally at
                # the center, it should very rarely go out of bound.
                if self._dc_estimator:
                    drift_shift = (self._dc_estimator.tot_drift[0] * sub_pxs[0],
                                   - self._dc_estimator.tot_drift[1] * sub_pxs[1])  # Y is upside down
                else:
                    drift_shift = (0, 0)  # m

                cspos = {"x": spos[0] - drift_shift[0],
                         "y": spos[1] - drift_shift[1]}
                if not (spos_rng[0] <= cspos["x"] <= spos_rng[2] and
                        spos_rng[1] <= cspos["y"] <= spos_rng[3]):
                    logging.error("Drift of %s px caused acquisition region out "
                                  "of bounds: needed to scan spot at %s.",
                                  drift_shift, cspos)
                    cspos = {"x": min(max(spos_rng[0], cspos["x"]), spos_rng[2]),
                             "y": min(max(spos_rng[1], cspos["y"]), spos_rng[3])}
                logging.debug("Scan stage pos: %s (including drift of %s)", cspos, drift_shift)

                # Remove unneeded moves, to not lose time with the actuator doing actually (almost) nothing
                for a, p in cspos.items():
                    if prev_spos[a] == p:
                        del cspos[a]

                sstage.moveAbsSync(cspos)
                prev_spos.update(cspos)
                logging.debug("Got stage synchronisation")

                failures = 0  # Keep track of synchronizing failures

                # acquire image
                while True:
                    start = time.time()
                    self._acq_min_date = start
                    for ce in self._acq_complete:
                        ce.clear()

                    if self._acq_state == CANCELLED:
                        raise CancelledError()

                    for s, sub in zip(self._streams[:-1], self._subscribers[:-1]):
                        s._dataflow.subscribe(sub)

                    time.sleep(0)  # give more chances spot has been already processed
                    self._trigger.notify()

                    # wait for detector to acquire image
                    timedout = self._waitForImage(px_time)

                    if self._acq_state == CANCELLED:
                        raise CancelledError()

                    # Check whether it went fine (= not too long and not too short)
                    dur = time.time() - start
                    if timedout or dur < px_time * 0.95:
                        if timedout:
                            # Note: it can happen we don't receive the data if there
                            # no more memory left (without any other warning).
                            # So we log the memory usage here too.
                            # TODO: Support also for Windows
                            import odemis.util.driver as udriver
                            memu = udriver.readMemoryUsage()
                            # Too bad, need to use VmSize to get any good value
                            logging.warning("Acquisition of repetition stream for "
                                            "pixel %s timed out after %g s. "
                                            "Memory usage is %d. Will try again",
                                            px_idx, px_time * 3 + 5, memu)
                        else:  # too fast to be possible (< the expected time - 5%)
                            logging.warning("Repetition stream acquisition took less than %g s: %g s, will try again",
                                            px_time, dur)
                        failures += 1
                        if failures >= 3:
                            # In three failures we just give up
                            raise IOError("Repetition stream acquisition repeatedly fails to synchronize")
                        else:
                            for s, sub, ad in zip(self._streams, self._subscribers, self._acq_data):
                                s._dataflow.unsubscribe(sub)
                                # Ensure we don't keep the data for this run
                                ad[:] = ad[:n]

                            # Restart the acquisition, hoping this time we will synchronize
                            # properly
                            time.sleep(1)
                            self._ccd_df.subscribe(self._subscribers[self._ccd_idx])
                            continue

                    # Normally, the SEM acquisitions have already completed
                    for s, sub, ce in zip(self._streams[:-1], self._subscribers[:-1], self._acq_complete[:-1]):
                        if not ce.wait(sem_time * 1.5 + 5):
                            raise TimeoutError("Acquisition of SEM pixel %s timed out after %g s"
                                               % (px_idx, sem_time * 1.5 + 5))
                        logging.debug("Got synchronisation from %s", s)
                        s._dataflow.unsubscribe(sub)

                    if self._acq_state == CANCELLED:
                        raise CancelledError()

                    # MD_POS default to the center of the sample stage, but it
                    # needs to be the position of the
                    #TODO: here different from _runAcquisitionEbeam
                    # sample stage + e-beam + scan stage translation (without the drift cor)
                    raw_pos = self._acq_data[0][-1].metadata[MD_POS]
                    strans = spos[0] - orig_spos["x"], spos[1] - orig_spos["y"]
                    cor_pos = raw_pos[0] + strans[0], raw_pos[1] + strans[1]
                    logging.debug("Updating pixel pos from %s to %s", raw_pos, cor_pos)
                    # In practice, for the e-beam data, it's only useful to
                    # update the metadata for the first pixel.
                    for adas in self._acq_data:
                        adas[-1].metadata[MD_POS] = cor_pos
                    ccd_data = self._acq_data[-1][-1]
                    self._acq_data[-1][-1] = self._preprocessData(len(self._streams), ccd_data, px_idx)
                    logging.debug("Processed CCD data %d = %s", n, px_idx)

                    n += 1
                    leech_time_left = (tot_num - n) * leech_time_ppx
                    self._updateProgress(future, time.time() - start, n, tot_num, leech_time_left)

                    # Check if it's time to run a leech
                    for li, l in enumerate(self.leeches):
                        if leech_np[li] is None:
                            continue
                        leech_np[li] -= 1
                        if leech_np[li] == 0:
                            # TODO: here different from _runAcquisitionEbeam
                            if isinstance(l, AnchorDriftCorrector):
                                # Move back to orig pos, to not compensate for the scan stage move
                                sstage.moveAbsSync(orig_spos)
                                prev_spos.update(orig_spos)
                            try:
                                np = l.next([d[-1] for d in self._acq_data])
                            except Exception:
                                logging.exception("Leech %s failed, will retry next pixel", l)
                                np = 1  # try again next pixel
                            leech_np[li] = np
                            if self._acq_state == CANCELLED:
                                raise CancelledError()

                    # Since we reached this point means everything went fine, so
                    # no need to retry
                    break

            # Done!
            for s, sub in zip(self._streams, self._subscribers):
                s._dataflow.unsubscribe(sub)
            self._ccd_df.synchronizedOn(None)

            with self._acq_lock:
                if self._acq_state == CANCELLED:
                    raise CancelledError()
                self._acq_state = FINISHED

            for n, das in enumerate(self._acq_data):
                self._onCompletedData(n, das)

            self._stopLeeches()

            if self._dc_estimator:
                self._anchor_raw.append(self._assembleAnchorData(self._dc_estimator.raw))

        except Exception as exp:
            if not isinstance(exp, CancelledError):
                logging.exception("Scan stage software sync acquisition of multiple detectors failed")

            # make sure it's all stopped
            for s, sub in zip(self._streams, self._subscribers):
                s._dataflow.unsubscribe(sub)
            self._ccd_df.synchronizedOn(None)

            self._raw = []
            self._anchor_raw = []
            if not isinstance(exp, CancelledError) and self._acq_state == CANCELLED:
                logging.warning("Converting exception to cancellation")
                raise CancelledError()
            raise
        else:
            return self.raw
        finally:
            if sstage:
                # Move back the stage to the center
                saxes = sstage.axes
                pos0 = {"x": sum(saxes["x"].range) / 2,
                        "y": sum(saxes["y"].range) / 2}
                sstage.moveAbs(pos0).result()

            for s in self._streams:
                s._unlinkHwVAs()
            self._acq_data = [[] for _ in self._streams]  # regain a bit of memory
            self._dc_estimator = None
            self._current_future = None
            self._acq_done.set()


class SEMMDStream(MultipleDetectorStream):
    """
    MDStream which handles when all the streams' detectors are linked to the
    e-beam.
    """

    def __init__(self, name, streams):
        """
        streams (List of Streams): All streams should be linked to the e-beam.
          The dwell time of the _last_ stream will be used as dwell time.
          Fuzzing is not supported, as it'd just mean software binning.
        """
        super(SEMMDStream, self).__init__(name, streams)
        for s in streams:
            if s._detector.role not in EBEAM_DETECTORS:
                raise ValueError("%s detector %s doesn't control e-beam" %
                                 (s, s._detector.name,))
        # Keep a link to the dwell time VA
        self._dwellTime = streams[-1]._getEmitterVA("dwellTime")

        # Checks that softwareTrigger is available
        if not isinstance(getattr(self._det0, "softwareTrigger", None),
                          model.EventBase):
            raise ValueError("%s detector has no softwareTrigger" % (self._det0.name,))
        self._trigger = self._det0.softwareTrigger

    def _estimateRawAcquisitionTime(self):
        """
        return (float): time in s for acquiring the whole image, without drift
         correction
        """
        # Each pixel x the dwell time (of the emitter) + 20% overhead
        dt = self._dwellTime.value
        duration = numpy.prod(self.repetition.value) * dt * 1.20
        # Add the setup time
        duration += self.SETUP_OVERHEAD

        return duration

    def _adjustHardwareSettings(self):
        """
        Read the SEM streams settings and adapt the SEM scanner accordingly.
        return (float): dwell time (for one pixel)
        """
        # Not much to do: dwell time is already set, and resolution will be set
        # dynamically.
        # We don't rely on the pixelSize from the RepetitionStream, because it's
        # used only for the GUI. Instead, recompute it based on the ROI and repetition.
        epxs = self._emitter.pixelSize.value
        pxs = self._getPixelSize()
        scale = (pxs / epxs[0], pxs / epxs[1])

        cscale = self._emitter.scale.clip(scale)
        if cscale != scale:
            logging.warning("Pixel size requested (%f m) < SEM pixel size (%f m)",
                            pxs, epxs[0])

        # TODO: check that no fuzzing is requested (as it's not supported and
        # not useful).

        self._emitter.scale.value = cscale
        return self._dwellTime.value

    def _runAcquisition(self, future):
        """
        Acquires images from the multiple detectors via software synchronisation.
        Warning: can be quite memory consuming if the grid is big
        returns (list of DataArray): all the data acquired
        raises:
          CancelledError() if cancelled
          Exceptions if error
        """
        try:
            self._acq_done.clear()
            px_time = self._adjustHardwareSettings()
            if self._emitter.dwellTime.value != px_time:
                raise IOError("Expected hw dt = %f but got %f" % (px_time, self._emitter.dwellTime.value))
            spot_pos = self._getSpotPositions()
            pos_flat = spot_pos.reshape((-1, 2))  # X/Y together (X iterates first)
            rep = self.repetition.value
            self._acq_data = [[] for _ in self._streams]  # just to be sure it's really empty
            self._raw = []
            self._anchor_raw = []
            logging.debug("Starting e-beam sync acquisition with components %s",
                          ", ".join(s._detector.name for s in self._streams))

            tot_num = numpy.prod(rep)

            # initialize leeches
            leech_np, leech_time_ppx = self._startLeeches(px_time, rep, tot_num)

            # number of spots scanned so far
            spots_sum = 0
            while spots_sum < tot_num:
                # Acquire the maximum amount of pixels until next leech
                npixels = min([np for np in leech_np if np is not None] +
                              [tot_num - spots_sum])  # max, in case of no leech
                n_y, n_x = leech.get_next_rectangle((rep[1], rep[0]), spots_sum, npixels)
                npixels = n_x * n_y
                # get_next_rectangle() takes care of always fitting in the
                # acquisition shape, even at the end.
                self._emitter.resolution.value = (n_x, n_y)

                # Move the beam to the center of the sub-frame
                trans = tuple(pos_flat[spots_sum:(spots_sum + npixels)].mean(axis=0))
                if self._dc_estimator:
                    trans = (trans[0] - self._dc_estimator.tot_drift[0],
                             trans[1] - self._dc_estimator.tot_drift[1])
                cptrans = self._emitter.translation.clip(trans)
                if cptrans != trans:
                    if self._dc_estimator:
                        logging.error("Drift of %s px caused acquisition region out "
                                      "of bounds: needed to scan spot at %s.",
                                      self._dc_estimator.tot_drift, trans)
                    else:
                        logging.error("Unexpected clipping in the scan spot position %s", trans)
                self._emitter.translation.value = cptrans

                spots_sum += npixels

                # and now the acquisition
                for ce in self._acq_complete:
                    ce.clear()

                self._df0.synchronizedOn(self._trigger)
                for s, sub in zip(self._streams, self._subscribers):
                    s._dataflow.subscribe(sub)
                start = time.time()
                self._acq_min_date = start
                self._trigger.notify()
                # Time to scan a frame
                frame_time = px_time * npixels

                # Wait for all the Dataflows to return the data. As all the
                # detectors are linked together to the e-beam, they should all
                # receive the data (almost) at the same time.
                max_end_t = start + frame_time * 10 + 5
                for i, s in enumerate(self._streams):
                    timeout = max(0.1, max_end_t - time.time())
                    if not self._acq_complete[i].wait(timeout):
                        raise TimeoutError("Acquisition of repetition stream for frame %s timed out after %g s"
                                           % (self._emitter.translation.value, time.time() - max_end_t))
                    if self._acq_state == CANCELLED:
                        raise CancelledError()
                    s._dataflow.unsubscribe(self._subscribers[i])

                # remove synchronisation
                self._df0.synchronizedOn(None)

                if self._acq_state == CANCELLED:
                    raise CancelledError()

                leech_time_left = (tot_num - spots_sum) * leech_time_ppx
                self._updateProgress(future, time.time() - start, spots_sum, tot_num, leech_time_left)

                # Check if it's time to run a leech
                for li, l in enumerate(self.leeches):
                    if leech_np[li] is None:
                        continue
                    leech_np[li] -= npixels
                    if leech_np[li] < 0:
                        logging.error("Acquired too many pixels, and skipped leech %s", l)
                        leech_np[li] = 0
                    if leech_np[li] == 0:
                        try:
                            np = l.next([d[-1] for d in self._acq_data])
                        except Exception:
                            logging.exception("Leech %s failed, will retry next pixel", l)
                            np = 1  # try again next pixel
                        leech_np[li] = np
                        if self._acq_state == CANCELLED:
                            raise CancelledError()

            # Done!
            with self._acq_lock:
                if self._acq_state == CANCELLED:
                    raise CancelledError()
                self._acq_state = FINISHED

            # TODO: ideally, we would directly assemble the _acq_data into the final
            # raw, which would avoid temporarily holding twice all the data.

            for n, das in enumerate(self._acq_data):
                self._onCompletedData(n, das)

            self._stopLeeches()

            if self._dc_estimator:
                self._anchor_raw.append(self._assembleAnchorData(self._dc_estimator.raw))

        except Exception as exp:
            if not isinstance(exp, CancelledError):
                logging.exception("Software sync acquisition of multiple detectors failed")

            # make sure it's all stopped
            for s, sub in zip(self._streams, self._subscribers):
                s._dataflow.unsubscribe(sub)
            self._df0.synchronizedOn(None)

            self._raw = []
            self._anchor_raw = []
            if not isinstance(exp, CancelledError) and self._acq_state == CANCELLED:
                logging.warning("Converting exception to cancellation", exc_info=True)
                raise CancelledError()
            raise
        else:
            return self.raw
        finally:
            for s in self._streams:
                s._unlinkHwVAs()
            self._acq_data = [[] for _ in self._streams]  # regain a bit of memory
            self._dc_estimator = None
            self._current_future = None
            self._acq_done.set()


class SEMSpectrumMDStream(SEMCCDMDStream):
    """
    Multiple detector Stream made of SEM + Spectrum.
    It handles acquisition, but not rendering (so .image always returns an empty
    image).
    """

    def _onCompletedData(self, n, raw_das):
        if n < len(self._streams) - 1:
            r = super(SEMSpectrumMDStream, self)._onCompletedData(n, raw_das)
            return r

        assert raw_das[0].shape[-2] == 1  # should be a spectra (Y == 1)

        # assemble all the CCD data into one
        rep = self.repetition.value
        spec_data = self._assembleSpecData(raw_das, rep)

        # Compute metadata based on SEM metadata
        sem_data = self._raw[0]  # _onCompletedData() should be called in order
        epxs = sem_data.metadata[MD_PIXEL_SIZE]
        # handle sub-pixels (aka fuzzing)
        tile_shape = self._emitter.resolution.value
#         shape_main = sem_data.shape[-1:-3:-1]  # 1,1,1,Y,X -> X, Y
#         rep = self.repetition.value
#         tile_shape = (shape_main[0] / rep[0], shape_main[1] / rep[1])
        pxs = (epxs[0] * tile_shape[0], epxs[1] * tile_shape[1])

        spec_data.metadata[MD_POS] = sem_data.metadata[MD_POS]
        spec_data.metadata[MD_PIXEL_SIZE] = pxs
        spec_data.metadata[MD_DESCRIPTION] = self._streams[n].name.value
        self._raw.append(spec_data)

    def _assembleSpecData(self, data_list, repetition):
        """
        Take all the data received from the spectrometer and assemble it in a
        cube.

        data_list (list of M DataArray of shape (1, N)): all the data received
        repetition (list of 2 int): X,Y shape of the high dimensions of the cube
         so that X * Y = M
        return (DataArray)
        """
        assert len(data_list) > 0

        # each element of acq_spect_buf has a shape of (1, N)
        # reshape to (N, 1)
        for e in data_list:
            e.shape = e.shape[::-1]
        # concatenate into one big array of (N, number of pixels)
        spec_data = numpy.concatenate(data_list, axis=1)
        # reshape to (C, 1, 1, Y, X) (as C must be the 5th dimension)
        spec_res = data_list[0].shape[0]
        spec_data.shape = (spec_res, 1, 1, repetition[1], repetition[0])

        # copy the metadata from the first point and add the ones from metadata
        md = data_list[0].metadata.copy()
        return model.DataArray(spec_data, metadata=md)


class SEMARMDStream(SEMCCDMDStream):
    """
    Multiple detector Stream made of SEM + AR.
    It handles acquisition, but not rendering (so .image always returns an empty
    image).
    """

    def _onCompletedData(self, n, raw_das):
        # raw_das: AR-images (arrays) excluding SEM-image (array)
        if n != self._ccd_idx:
            return super(SEMARMDStream, self)._onCompletedData(n, raw_das)

        # Not much to do: just save everything as is

        # MD_AR_POLE is set automatically, copied from the lens property.
        # In theory it's dependent on MD_POS, but so slightly that we don't need
        # to correct it.
        sname = self._streams[n].name.value
        for d in raw_das:
            d.metadata[MD_DESCRIPTION] = sname

        if len(raw_das) != numpy.prod(self.repetition.value):
            logging.error("Only got %d AR acquisitions while expected %d",
                          len(raw_das), numpy.prod(self.repetition.value))

        # self._raw: first entry is sem-array, second onwards is ar-arrays
        self._raw.extend(raw_das)  # append ar arrays


class MomentOfInertiaMDStream(SEMCCDMDStream):
    """
    Multiple detector Stream made of SEM + CCD, with direct computation of the
    moment of inertia (MoI) and spot size of the CCD images. The MoI is
    assembled into one big image for the CCD.
    Used by the MomentOfInertiaLiveStream to provide display in the mirror
    alignment mode for SPARCv2.
    .raw actually contains: SEM data, moment of inertia, valid array, spot intensity at center (array of 0 dim)
    """

    def __init__(self, name, streams):
        super(MomentOfInertiaMDStream, self).__init__(name, streams)

        # Region of interest as left, top, right, bottom (in ratio from the
        # whole area of the emitter => between 0 and 1) that defines the region
        # to be acquired for the MoI computation.
        # This is expected to be centered to the lens pole position.
        self.detROI = model.TupleContinuous((0, 0, 1, 1),
                                            range=((0, 0, 0, 0), (1, 1, 1, 1)),
                                            cls=(int, long, float))

        self.background = model.VigilantAttribute(None)  # None or 2D DataArray

        self._center_image_i = (0, 0)  # iteration at the center (for spot size)
        self._center_raw = None  # raw data at the center

        # For computing the moment of inertia in background
        self._executor = None

    def _adjustHardwareSettings(self):
        """
        Set the CCD settings to crop the FoV around the pole position to
        optimize the speed of the MoI computation.
        return (float): estimated time for a whole CCD image
        """
        # We should remove res setting from the GUI when this ROI is used.
        roi = self.detROI.value
        center = ((roi[0] + roi[2]) / 2, (roi[1] + roi[3]) / 2)
        width = (roi[2] - roi[0], roi[3] - roi[1])

        if not self._ccd.resolution.read_only:
            shape = self._ccd.shape
            binning = self._ccd.binning.value
            res = (max(1, int(round(shape[0] * width[0] / binning[0]))),
                   max(1, int(round(shape[1] * width[1] / binning[1]))))
            # translation is distance from center (situated at 0.5, 0.5), can be floats
            trans = (shape[0] * (center[0] - 0.5), shape[1] * (center[1] - 0.5))
            # clip translation so ROI remains in bounds
            bin_trans = (trans[0] / binning[0], trans[1] / binning[1])
            half_res = (int(round(res[0] / 2)), int(round(res[1] / 2)))
            cur_res = (shape[0] / binning[0], shape[1] / binning[1])
            bin_trans = (numpy.clip(bin_trans[0], -(cur_res[0] / 2) + half_res[0], (cur_res[0] / 2) - half_res[0]),
                         numpy.clip(bin_trans[1], -(cur_res[1] / 2) + half_res[1], (cur_res[1] / 2) - half_res[1]))
            trans = (int(bin_trans[0] * binning[0]), int(bin_trans[1] * binning[1]))
            # always in this order
            self._ccd.resolution.value = self._ccd.resolution.clip(res)

            if model.hasVA(self._ccd, "translation"):
                self._ccd.translation.value = trans
            else:
                logging.info("CCD doesn't support ROI translation, would have used %s", trans)
        return super(MomentOfInertiaMDStream, self)._adjustHardwareSettings()

    def acquire(self):
        if self._current_future is not None and not self._current_future.done():
            raise IOError("Cannot do multiple acquisitions simultaneously")

        # Reset some data
        self._center_image_i = tuple((v - 1) // 2 for v in self.repetition.value)
        self._center_raw = None

        return super(MomentOfInertiaMDStream, self).acquire()

    def _runAcquisition(self, future):
        # TODO: More than one thread useful? Use processes instead? + based on number of CPUs
        self._executor = futures.ThreadPoolExecutor(2)
        try:
            return super(MomentOfInertiaMDStream, self)._runAcquisition(future)
        finally:
            # We don't need futures anymore
            self._executor.shutdown(wait=False)

    def _preprocessData(self, n, data, i):
        """
        return (Future)
        """
        if n < len(self._streams) - 1:
            return super(MomentOfInertiaMDStream, self)._preprocessData(n, data, i)

        # Instead of storing the actual data, we queue the MoI computation in a future
        logging.debug("Queueing MoI computation")

        if i == (0, 0):
            # No need to calculate the drange every time:
            self._drange = img.guessDRange(data)

        # Compute spot size only for the center image
        ss = (i == self._center_image_i)
        if i == self._center_image_i:
            self._center_raw = data

        return self._executor.submit(self.ComputeMoI, data, self.background.value, self._drange, ss)

    def _onCompletedData(self, n, raw_das):
        if n < len(self._streams) - 1:
            return super(MomentOfInertiaMDStream, self)._onCompletedData(n, raw_das)

        # Wait for the moment of inertia calculation results
        mi_results = []
        valid_results = []
        spot_size = None
        for f in raw_das:
            mi, valid, ss = f.result()
            if ss is not None:
                spot_size = ss
            mi_results.append(mi)
            valid_results.append(valid)

        # Get position based on SEM metadata
        center, pxs = self._get_center_pxs(self.repetition.value, (1, 1),
                                           self._acq_data[0][0])
        md = {MD_POS: center,
              MD_PIXEL_SIZE: pxs}

        # convert the list into array
        moi_array = numpy.array(mi_results)
        moi_array.shape = self.repetition.value
        self._raw.append(model.DataArray(moi_array, md))

        valid_array = numpy.array(valid_results)
        valid_array.shape = self.repetition.value
        self._raw.append(model.DataArray(valid_array, md))

        self._raw.append(model.DataArray(spot_size))
        self._raw.append(self._center_raw)

    def ComputeMoI(self, data, background, drange, spot_size=False):
        """
        It performs the moment of inertia calculation (and a bit more)
        data (model.DataArray): The AR optical image
        background (None or model.DataArray): Background image that we use for subtraction
        drange (tuple of floats): drange of data
        spot_size (bool): if True also calculate the spot size
        returns:
           moi (float): moment of inertia
           valid (bool): False if some pixels are clipped (which probably means
             the computed moment of inertia is invalid) or MoI cannot be computed
             (eg, the image is fully black).
           spot size (None or float): spot size if was asked, otherwise None
        """
        logging.debug("Moment of inertia calculation...")

        try:
            moment_of_inertia = spot.MomentOfInertia(data, background)
#             moment_of_inertia += random.uniform(0, 10)  # DEBUG
#             if random.randint(0, 10) == 0:  # DEBUG
#                 moment_of_inertia = float("NaN")
            valid = not img.isClipping(data, drange) and not math.isnan(moment_of_inertia)
            # valid = random.choice((True, False))  # DEBUG
            if spot_size:
                spot_estimation = spot.SpotIntensity(data, background)
            else:
                spot_estimation = None
            return moment_of_inertia, valid, spot_estimation
        except Exception:
            # This is a future running in a future... a pain to get the traceback
            # in case of exception, so drop it immediately on the log too
            logging.exception("Failure to compute moment of inertia")
            raise


# TODO: ideally it should inherit from FluoStream
class ScannedFluoMDStream(MultipleDetectorStream):
    """
    Stream to acquire multiple ScannedFluoStreams simultaneously
    """
    def __init__(self, name, streams):
        """
        streams (list of ScannedFluoStreams): they should all have the same scanner
          and emitter (just a different detector). At least one stream should be
          provided.
        """
        super(ScannedFluoMDStream, self).__init__(name, streams)

        self._setting_stream = self._s0.setting_stream
        for s in streams[1:]:
            assert self._s0.scanner == s.scanner
            assert self._setting_stream == s.setting_stream

        self._trigger = self._det0.softwareTrigger

    @property
    def raw(self):
        # We can use the .raw of the substreams, as the live streams are the same format
        r = []
        for s in self._streams:
            r.extend(s.raw)

        return r

    # Methods required by MultipleDetectorStream
    def _estimateRawAcquisitionTime(self):
        """
        return (float): time in s for acquiring the whole image, without drift
         correction
        """
        # It takes the same time as just one stream
        return self.streams[0].estimateAcquisitionTime()

    def estimateAcquisitionTime(self):
        # No drift correction supported => easy
        return self._estimateRawAcquisitionTime()

    def _onCompletedData(self, n, raw_das):
        """
        Called at the end of an entire acquisition. It should assemble the data
        and append it to ._raw .
        Override if you need to process the data in a different way.
        n (0<=int): the detector/stream index
        raw_das (list of DataArray): data as received from the detector.
           The data is ordered, with X changing fast, then Y slow
        """
        # explicitly add names to make sure they are different
        da = raw_das[0]
        da.metadata[MD_DESCRIPTION] = self._streams[n].name.value
        # Not adding to the _raw, as it's kept on the streams directly

    def _adjustHardwareSettings(self):
        """
        Adapt the emitter/scanner/detector settings.
        return (float): estimated time per acquisition
        """
        # TODO: all the linkHwVAs should happen here
        if self._setting_stream:
            self._setting_stream.is_active.value = True

        # All streams have the same excitation, so do it only once
        self._streams[0]._setup_excitation()
        for s in self._streams:
            s._setup_emission()

        return self.estimateAcquisitionTime()

    def _cancelAcquisition(self, future):
        with self._acq_lock:
            if self._acq_state == FINISHED:
                return False  # too late
            self._acq_state = CANCELLED

        logging.debug("Cancelling acquisition of components %s and %s",
                      self._streams[0].emitter.name,
                      self._streams[0].scanner.name)

        # set the events, so the acq thread doesn't wait for them
        for i in range(len(self._streams)):
            self._acq_complete[i].set()
        self._streams[0]._dataflow.synchronizedOn(None)

        # Wait for the thread to be complete (and hardware state restored)
        self._acq_done.wait(5)
        return True

    def _onData(self, n, df, data):
        logging.debug("Stream %d data received", n)
        s = self._streams[n]
        if self._acq_min_date > data.metadata.get(model.MD_ACQ_DATE, 0):
            # This is a sign that the e-beam might have been at the wrong (old)
            # position while Rep data is acquiring
            logging.warning("Dropping data because it seems started %g s too early",
                            self._acq_min_date - data.metadata.get(model.MD_ACQ_DATE, 0))
            if n == 0:
                # As the first detector is synchronised, we need to restart it
                # TODO: probably not necessary, as the typical reason it arrived
                # early is that the detectors were already running, in which case
                # they haven't "consumed" the previous trigger yet
                self._trigger.notify()
            return

        if not self._acq_complete[n].is_set():
            s._onNewData(s._dataflow, data)
            self._acq_complete[n].set()
            # TODO: unsubscribe here?

    def _runAcquisition(self, future):
        """
        Acquires images from the multiple detectors via software synchronisation.
        Warning: can be quite memory consuming if the grid is big
        returns (list of DataArray): all the data acquired
        raises:
          CancelledError() if cancelled
          Exceptions if error
        """
        try:
            self._acq_done.clear()
            acq_time = self._adjustHardwareSettings()

            # Synchronise one detector, so that it's possible to subscribe without
            # the acquisition immediately starting. Once all the detectors are
            # subscribed, we'll notify the detector and it will start.
            self._df0.synchronizedOn(self._trigger)
            for s in self.streams[1:]:
                s._dataflow.synchronizedOn(None)  # Just to be sure

            subscribers = []  # to keep a ref
            for i, s in enumerate(self._streams):
                p_subscriber = partial(self._onData, i)
                subscribers.append(p_subscriber)
                s._dataflow.subscribe(p_subscriber)
                self._acq_complete[i].clear()

            if self._acq_state == CANCELLED:
                raise CancelledError()

            self._acq_min_date = time.time()
            self._trigger.notify()
            logging.debug("Starting confocal acquisition")

            # TODO: immediately remove the synchronisation? It's not needed after
            # the start.

            # Wait until all the data is received
            for i, s in enumerate(self._streams):
                # TODO: It should arrive at the same time, so after the first stream less timeout
                if not self._acq_complete[i].wait(3 + acq_time * 1.5):
                    raise IOError("Confocal acquisition hasn't received data after %g s" %
                                  (time.time() - self._acq_min_date,))
                if self._acq_state == CANCELLED:
                    raise CancelledError()
                s._dataflow.unsubscribe(subscribers[i])
                s._dataflow.synchronizedOn(None)  # Just to be sure

            # Done
            self._streams[0]._stop_light()
            logging.debug("All confocal acquisition data received")
            for n, s in enumerate(self._streams):
                self._onCompletedData(n, s.raw)

        except Exception as exp:
            if not isinstance(exp, CancelledError):
                logging.exception("Acquisition of confocal multiple detectors failed")
            else:
                logging.debug("Confocal acquisition cancelled")

            self._streams[0]._stop_light()
            for i, s in enumerate(self._streams):
                s._dataflow.unsubscribe(subscribers[i])
                s._dataflow.synchronizedOn(None)  # Just to be sure

            if not isinstance(exp, CancelledError) and self._acq_state == CANCELLED:
                logging.warning("Converting exception to cancellation")
                raise CancelledError()
            raise
        else:
            return self.raw
        finally:
            for s in self._streams:
                s._unlinkHwVAs()
            if self._setting_stream:
                self._setting_stream.is_active.value = False
            self._current_future = None
            self._acq_done.set()


class ScannedRemoteTCStream(LiveStream):
    
    def __init__(self, name, stream, **kwargs):
        '''
        A stream aimed at FLIM acquisition with a time-correlator on a SECOM.
        It acquires by scanning the defined ROI multiple times. Each scanned
        frame uses the maximum dwell time accepted by the scanner and is
        accumulated until the stream.dwellTime is reached.

        It acquires simultaneously the "rough" data received via the tc_detector
        (eg, an APD) synchronised with the laser-mirror and the actual FLIM data
        acquired by the time-correlator. For now, we expect the time-correlator
        to store the data separately, which is why it's not returned by the
        stream.

        During the acquisition, it updates .image with the latest raw data.

        stream: (ScannedTCSettingsStream) contains all necessary devices as children
        '''
        # We don't use the stream.detector because it's the tc_detector, which
        # we only use for synchronizing the scanner and recording the basic data,
        # or it's the tc_detector_live, which we don't use at all.
        super(ScannedRemoteTCStream, self).__init__(name, stream.time_correlator,
            stream.time_correlator.dataflow, stream.emitter, opm=stream._opm, **kwargs)

        # Retrieve devices from the helper stream
        self._stream = stream
        # self._emitter = stream.emitter  # =.emitter
        # self._time_correlator = stream.time_correlator  # =.detector
        self._scanner = stream.scanner
        self._tc_detector = stream.tc_detector  # synchronize to the scanner
        self._tc_scanner = stream.tc_scanner  # might be None

        # the total dwell time
        self._dwellTime = stream.dwellTime
        self.roi = stream.roi
        self.repetition = stream.repetition
        if self._tc_scanner and hasVA(self._tc_scanner, "filename"):
            self.filename = self._tc_scanner.filename

        # For the acquisition
        self._acq_lock = threading.Lock()
        self._acq_state = RUNNING
        self._acq_thread = None  # thread
        self._prog_sum = 0  # s, for progress time estimation
        self._data_queue = Queue.Queue()
        self._current_future = None

    def acquire(self):
        # Make sure every stream is prepared, not really necessary to check _prepared
        f = self.prepare()
        f.result()

        # TODO: if already acquiring, queue the Future for later acquisition
        if self._current_future is not None and not self._current_future.done():
            raise IOError("Cannot do multiple acquisitions simultaneously")

        if self._acq_thread and self._acq_thread.isAlive():
            logging.debug("Waiting for previous acquisition to fully finish")
            self._acq_thread.join(10)
            if self._acq_thread.isAlive():
                logging.error("Previous acquisition not ending, will acquire anyway")

        est_start = time.time() + 0.1
        f = model.ProgressiveFuture(start=est_start,
                                    end=est_start + self.estimateAcquisitionTime())
        self._current_future = f
        self._acq_state = RUNNING  # TODO: move to per acquisition
        self._prog_sum = 0
        f.task_canceller = self._cancelAcquisition

        # run task in separate thread
        self._acq_thread = executeAsyncTask(f, self._runAcquisition, args=(f,))
        return f

    def _prepareHardware(self):
        '''
        Prepare hardware for acquisition and return the best pixel dwelltime value
        '''

        scale, res, trans = self._computeROISettings(self._stream.roi.value)

        # always in this order
        self._scanner.scale.value = scale
        self._scanner.resolution.value = res
        self._scanner.translation.value = trans

        logging.debug("Scanner set to scale %s, res %s, trans %s",
                      self._scanner.scale.value,
                      self._scanner.resolution.value,
                      self._scanner.translation.value)

        # The dwell time from the Nikon C2 will set based on what the device is capable of
        # As a result, we need to recalculate our total dwell time based around this value
        # and the number of frames we can compute
        px_dt = min(self._dwellTime.value, self._scanner.dwellTime.range[1])
        self._scanner.dwellTime.value = px_dt
        px_dt = self._scanner.dwellTime.value
        nfr = int(math.ceil(self._dwellTime.value / px_dt))  # number of frames
        px_dt = self._dwellTime.value / nfr  # the new dwell time per frame (slightly shorter than we asked before)
        self._scanner.dwellTime.value = px_dt  # try set the C2 dwell time value again.
        logging.info("Total dwell time: %f s, Pixel Dwell time: %f, Resolution: %s, collecting %d frames...",
                     self._dwellTime.value, px_dt, self._scanner.resolution.value, nfr)

        if self._tc_scanner and hasVA(self._tc_scanner, "dwellTime"):
            self._tc_scanner.dwellTime.value = self._dwellTime.value

        return px_dt, nfr

    def _setEmission(self, value):
        # set all light emissions at once to a value
        em = self._emitter.emissions.value
        em = [value] * len(em)
        self._emitter.emissions.value = em

    def _computeROISettings(self, roi):
        """
        roi (4 0<=floats<=1)
        return:
            scale (2 ints)
            res (2 ints)
            trans (2 floats)
        """
        # We should remove res setting from the GUI when this ROI is used.
        center = ((roi[0] + roi[2]) / 2, (roi[1] + roi[3]) / 2)
        width = (roi[2] - roi[0], roi[3] - roi[1])

        shape = self._scanner.shape
        # translation is distance from center (situated at 0.5, 0.5), can be floats
        trans = (shape[0] * (center[0] - 0.5), shape[1] * (center[1] - 0.5))
        res = self.repetition.value
        scale = (width[0] * shape[0] / res[0], width[1] * shape[1] / res[1])

        return scale, res, trans

    def _runAcquisition(self, future):
        logging.debug("Starting FLIM acquisition")
        try:
            self._stream._linkHwVAs()
            self.raw = []
            assert self._data_queue.empty()

            px_dt, nfr = self._prepareHardware()
            frame_time = px_dt * numpy.prod(self._scanner.resolution.value)
            logging.info("Theoretical minimum frame time: %f s", frame_time)

            # Start Symphotime acquisition
            self._detector.data.subscribe(self._onAcqStop)

            # Turn on the lights
            self._setEmission(1)

            # Start the acquisition
            self._tc_detector.data.subscribe(self._onNewData)

            # For each frame
            for i in range(nfr):
                # wait for the next raw frame
                logging.info("Getting frame %d/%d", i + 1, nfr)
                tstart = time.time()
                ttimeout = tstart + frame_time * 5 + 1

                while True:  # until the frame arrives, timed out, or cancelled
                    if self._acq_state == CANCELLED:
                        raise CancelledError("Acquisition canceled")

                    try:
                        data = self._data_queue.get(timeout=0.1)
                    except Queue.Empty:
                        if time.time() > ttimeout:
                            raise IOError("Timed out waiting for frame, after %f s" % (time.time() - tstart,))
                        continue  # will check again whether the acquisition is cancelled
                    break  # data has been received

                # Got the frame -> accumulate it
                dur = time.time() - tstart
                logging.debug("Received frame %d after %f s", i, dur)
                self._add_frame(data)
                self._updateProgress(future, dur, i + 1, nfr)

            # Acquisition completed
            logging.debug("FLIM acquisition completed successfully.")

        except CancelledError:
            logging.info("Acquisition cancelled")
            self._acq_state = CANCELLED
            raise
        except Exception:
            logging.exception("Failure during ScannedTC acquisition")
            raise
        finally:
            logging.debug("FLIM acquisition ended")

            # Ensure all the detectors are stopped
            self._tc_detector.data.unsubscribe(self._onNewData)
            self._detector.data.unsubscribe(self._onAcqStop)
            self._stream._unlinkHwVAs()
            # turn off the light
            self._setEmission(0)

            # If cancelled, some data might still be queued => forget about it
            self._data_queue = Queue.Queue()

            with self._acq_lock:
                if self._acq_state == CANCELLED:
                    raise CancelledError()
                self._acq_state = FINISHED

        return self.raw

    def _onAcqStop(self, dataflow, data):
        pass

    def _add_frame(self, data):
        """
        Accumulate the raw frame to update the .raw
        data (DataArray): the new raw frame
        """
        logging.debug("Adding frame of shape %s and type %s", data.shape, data.dtype)
        if not self.raw:
            # TODO: be more careful with the dtype
            data = data.astype(numpy.uint32)
            self.raw = [data]

            # Force update histogram to ensure it exists.
            self._updateHistogram(data)
        else:
            if self.raw[0].shape == data.shape:
                self.raw[0] += data  # Uses numpy element-wise addition
                self.raw[0].metadata[MD_DWELL_TIME] += data.metadata[MD_DWELL_TIME]
            else:
                logging.error("New data array from tc-detector has different shape %s from previous one, can't accumulate data",
                               data.shape)

            self._shouldUpdateHistogram()

        self._shouldUpdateImage()

    def _updateProgress(self, future, dur, current, tot, bonus=2):
        """
        update end time of future by indicating the time for one new pixel
        future (ProgressiveFuture): future to update
        dur (float): time it took to do this acquisition
        current (1<int<=tot): current number of acquisitions done
        tot (0<int): number of acquisitions
        bonus (0<float): additional time needed (eg, for leeches)
        """
        # Trick: we don't count the first frame because it's often
        # much slower and so messes up the estimation
        if current <= 1:
            return

        self._prog_sum += dur
        ratio = (tot - current) / (current - 1)
        left = self._prog_sum * ratio
        time_assemble = 0.001 * tot  # very rough approximation
        # add some overhead for the end of the acquisition
        tot_left = left + time_assemble + bonus + 0.1
        tot_left *= 1.1  # extra padding
        future.set_progress(end=time.time() + tot_left)

    def _onNewData(self, dataflow, data):
        # Add frame data to the queue for processing later.
        # This way, if the frame time is very fast, we will not miss frames.
        logging.debug("New data received of shape %s", data.shape)
        self._data_queue.put(data)

    def _cancelAcquisition(self, future):
        with self._acq_lock:
            if self._acq_state == FINISHED:
                return False  # too late
            self._acq_state = CANCELLED

        logging.debug("Cancelling acquisition of components %s and %s",
                      self._tc_detector.name, self._detector.name)

        # Wait for the thread to be complete (and hardware state restored)
        if self._acq_thread:
            self._acq_thread.join(5)

        return True

    def estimateAcquisitionTime(self):
        return self._dwellTime.value * numpy.prod(self.repetition.value) * 1.2 + 1.0

