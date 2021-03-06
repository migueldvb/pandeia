# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import division, absolute_import

import numpy as np
import numpy.ma as ma
import copy
import astropy.io.fits as fits

from .coords import Grid, IrregularGrid
from .custom_exceptions import EngineOutputError
from six.moves import range
from six.moves import zip


class Report(object):

    """
    This class contains functionality for generating reports and data products from the results
    of an ETC calculation.

    Parameters
    ----------
    calc_input: dict
        The engine API format dict used to configure the calculation
    signal_list : list of etc3D.DetectorSignal instances
        The calculated signal on the detector
    noise_list : list of etc3D.DetectorNoise instances
        The calculated noise on the detector
    extracted : dict
        The extracted data product generated by the Strategy
    warnings: dict
        Dict of warnings produced and collected through the course of the calculation
    """

    def __init__(self, calc_input, signal_list, noise_list, saturation_list, extracted, warnings):
        # For single pointing calculations, the signal and noise lists have one element.
        # However, handling calculations that involve multiple pointings is more tricky.
        # We want to grab one of the on_target calculations to get the relevant target
        # information for the report.
        if not isinstance(signal_list, list):
            raise EngineOutputError(value="Must provide Report signals and noises as lists.")

        self.extracted = extracted

        self.sub_reports = []
        if len(signal_list) == 1:
            self.signal = signal_list[0]
            self.noise = noise_list[0]
            self.saturation = saturation_list[0]
        elif len(signal_list) > 1:
            # Fish out the list of which ones are targets.
            """
            on_target = calc_input['strategy']['on_target']  # True if target, False otherwise
            target = [i for i, v in enumerate(on_target) if v][0]  # get first target
            """
            # The above method does not work in general. Assume the first plane is a target and
            # refactor when implementing dithers in general.
            target = 0

            self.signal = signal_list[target]
            self.noise = noise_list[target]
            self.saturation = saturation_list[target]
            for signal, noise, saturation in zip(signal_list, noise_list, saturation_list):
                r = Report(calc_input, [signal], [noise], [saturation], extracted, warnings)
                self.sub_reports.append(r)

        # We use the calculation inputs for a few things.
        self.input = calc_input

        # Get the full Signal and Noise objects for inspection.
        self.signals = signal_list
        self.noises = noise_list
        self.saturations = saturation_list

        # get the ExposureSpecification.
        self.exposure_specification = self.signal.current_instrument.get_exposure_pars()

        # get the type of observation. basically, this is the format of the detector plane data.
        # it will be either 'image' (X spatial, Y spatial) or 'spec' (X wavelength, Y spatial)
        self.projection_type = self.signal.projection_type

        # Target spectrum
        target_curve = [self.signal.wave, self.signal.total_flux]

        # Transmission/focal plane rate
        trans_curve = [self.signal.wave, self.signal.fp_rate.sum(axis=(0, 1))]

        # input background
        bg_curve = [self.signal.wave, self.signal.background.MJy_sr]

        # Background rate
        bg_rate_curve = [self.signal.wave, self.signal.bg_fp_rate]

        # this is the wavelength sampling of the calculation
        self.wave = self.signal.wave

        # this is the data cube of the input signal
        self.flux = self.signal.flux_cube_list

        # this is the data cube of the input signal plus background
        self.flux_plus_bg = self.signal.flux_plus_bg_list

        # This is the background rate in each pixel without sources
        self.bg_pix = self.signal.rate_plus_bg - self.signal.rate

        # Signal and noise in 2D. Get from extracted products
        s = extracted['detector_signal']
        n = extracted['detector_noise']

        # get areas in pixels of source and background regions
        self.extraction_area = extracted['extraction_area']
        if extracted['background_area'] is not None:
            self.background_area = extracted['background_area']
        else:
            self.background_area = None

        # Check for full saturation and set the noise to infinity accordingly. This is necessary since
        # the engine currently returns noise=0 if the pixel has full saturation, which is confusing since the
        # noise is not 0, but rather undertermined or infinite. Setting the noise to NaN ensures that the S/N
        # of saturated pixels are NaNs.
        n[self.saturation == 2] = np.nan
        self.detector_sn = (s - self.bg_pix) / n
        self.detector_signal = s + n * np.random.randn(n.shape[0], n.shape[1])

        # this is the spatial grid for the calculation
        self.grid = self.signal.grid

        # this is the detector plane pixel grid. for most modes we just grab and use it directly.
        # however, slitless we want to redefine it to be spatial on both axes.
        if self.signal.projection_type in ('slitless'):
            orig_grid = self.signal.pixgrid_list[0]
            # detector plane gets rotated depending on dispersion_axis so adjust accordingly
            if self.signal.dispersion_axis == 'x':
                self.pix_grid = Grid(self.grid.xsamp, orig_grid.ysamp, orig_grid.nx, orig_grid.ny)
            else:
                self.pix_grid = Grid(self.grid.ysamp, orig_grid.xsamp, orig_grid.ny, orig_grid.nx)
        elif self.input['strategy']['method'] in ('ifuapphot', 'ifunodinscene', 'ifunodoffscene'):
            # for IFUs we want the reconstructed image plane that's build by the strategy
            self.pix_grid = extracted['plane_grid']
        elif self.signal.projection_type in ('multiorder'):
            orig_grid = self.signal.pixgrid_list[0]
            self.pix_grid = IrregularGrid(np.arange(orig_grid.nx), np.arange(orig_grid.ny))
        else:
            self.pix_grid = self.signal.pixgrid_list[0]

        # Values calculated by the strategy
        sn = extracted['extracted_flux'] / extracted['extracted_noise']
        if self.signal.projection_type == 'image':
            # this is the wavelength sampling on the detector. in imaging mode this is
            # the effective wavelength of the filter + detector + optics
            self.wave_pix = self.signal.wave_pix  # convert to real np.array of len = 1
        elif self.signal.projection_type == 'spec':
            # this is the wavelength sampling on the detector.
            self.wave_pix = extracted['wavelength']  # this is already a 1D np.array
            self.cube_signal, self.cube_noise, self.cube_saturation, self.cube_plane_grid = extracted['reconstructed']
            self.cube_sim = self.cube_signal + self.cube_noise * np.random.randn(*self.cube_noise.shape)
        elif self.signal.projection_type == 'slitless':
            self.wave_pix = extracted['wavelength']
            self.detector_sn_unrot = self.detector_sn
            self.detector_signal_unrot = self.detector_signal
            self.saturation_unrot = self.saturation
            if self.signal.dispersion_axis == 'y':
                # need to rotate these 90 deg clockwise to match our normal axis orientation. np.rot90 only works CCW
                # so need to flip, rotate, and then flip back. note that currently this case implies that dispersion
                # axis is 90 degrees.
                self.wave_pix = self.wave_pix[::-1]
                self.detector_sn = np.flipud(np.rot90(np.flipud(self.detector_sn)))
                self.detector_signal = np.flipud(np.rot90(np.flipud(self.detector_signal)))
                self.saturation = np.flipud(np.rot90(np.flipud(self.saturation)))
        elif self.signal.projection_type == 'multiorder':
            self.wave_pix = extracted['wavelength']
            self.detector_sn_unrot = self.detector_sn
            self.detector_signal_unrot = self.detector_signal
            self.saturation_unrot = self.saturation
            self.detector_sn = np.rot90(self.detector_sn)
            self.detector_signal = np.rot90(self.detector_signal)
            self.saturation = np.rot90(self.saturation)
        else:
            raise EngineOutputError(value="Unsupported projection_type: %s" % self.signal.projection_type)

        sn_curve = [self.wave_pix, sn]
        extracted_noise = [self.wave_pix, extracted['extracted_noise']]
        extracted_flux = [self.wave_pix, extracted['extracted_flux']]
        extracted_flux_plus_bg = [self.wave_pix, extracted['extracted_flux_plus_bg']]
        
        total_flux = [self.wave_pix, extracted['source_flux_in_fov']]
        self.curves = {
            'target': target_curve,
            'fp': trans_curve,
            'bg': bg_curve,
            'bg_rate': bg_rate_curve,
            'sn': sn_curve,
            'extracted_noise': extracted_noise,
            'extracted_flux': extracted_flux,
            'extracted_flux_plus_bg': extracted_flux_plus_bg,
            'total_flux': total_flux,
            'extracted_bg_total': [self.wave_pix, extracted['extracted_bg_total']],
            'extracted_bg_only': [self.wave_pix, extracted['extracted_bg_only']],
            'extracted_contamination': [self.wave_pix, extracted['extracted_bg_total'] - extracted['extracted_bg_only']],
            'n_partial_saturated': [self.wave_pix,extracted['saturation_products']['partial']],
            'n_full_saturated': [self.wave_pix,extracted['saturation_products']['full']]
        }
        # If there is a contrast product, add it
        if 'contrast_curve' in extracted:
            self.curves['contrast'] = extracted['contrast_curve']

        self.warnings = warnings

    def as_dict(self):
        """
        Produce report in dictionary format conformant with the engine API.

        Most strategies operate on and extract flux, but some (eg TACentroid) use the same machinery, which is
        in principle completely general, to extract a properly weighted sum of something else (eg, centroid
        for TACentroid).  These strategies, the "extracted_flux" is the extracted product of interest, and
        the "extracted_noise" is the standard deviation of the extracted product.

        Returns
        -------
        r: dict
            Dictionary containing the results of the calculation, associated information,
            and any applicable warnings.
        """
        # fill out report with the calculation outputs...
        r = {}

        # look for any sub-reports and convert them
        r['sub_reports'] = []
        for report in self.sub_reports:
            r['sub_reports'].append(report.as_dict())

        # set up the coordinate transform information that describes the image axes
        # for the 2D and 3D data.  first, get the spatial information from self.Grid
        r['transform'] = self.pix_grid.as_dict()

        # the wavelength sampling of the input spectrum is different
        # then the sampling by the detector. the model cubes that are
        # used to make the calculation use the input sampling. the detector
        # plane outputs for spectroscopy and IFU use the detector sampling

        # ## input model wavelength sampling ## #
        r['transform'].update(self.signal.spectral_model_transform())

        # ## detector plane wavelength sampling ## #
        r['transform'].update(self.signal.spectral_detector_transform())

        # the spectral mapping can be shifted by strategies (e.g. slitless). this is
        # reflected in wave_pix so update from there.
        r['transform']['wave_det_min'] = self.wave_pix.min()
        r['transform']['wave_det_max'] = self.wave_pix.max()
        r['transform']['wave_det_refval'] = self.wave_pix[0]

        # scalar data products
        r['scalar'] = {}
        # if there's a detector gap, the noise is 0 and sn goes to np.inf. fix that here
        # so that we can get a proper maximum S/N and report scalar values there.
        # NOTE: this value is only used in the scalar values, the 1d sn
        # may still contain a NaN.
        sn = ma.fix_invalid(self.curves['sn'][1], fill_value=0.0).data

        if self.input['strategy']['method'] in [
            'soss',
            'specapphot',
            'msafullapphot',
            'ifuapphot',
            'ifunodinscene',
            'ifunodoffscene'
        ]:
            wave_index = int(len(self.wave_pix) / 2.0)
            if 'reference_wavelength' in self.input['strategy']:
                wref = self.input['strategy']['reference_wavelength']
                if wref is not None:
                    if wref >= self.wave_pix.min() and wref <= self.wave_pix.max():
                        wave_index = (np.abs(self.wave_pix - wref)).argmin()
                    else:
                        self.warnings['bad_waveref'] = "Specified wavelength, %f, out of range [%f, %f]. " % (
                            wref,
                            self.wave_pix.min(),
                            self.wave_pix.max()
                        )
                        self.warnings['bad_waveref'] += "Using %f to select diagnostic planes instead." % (
                            float(self.wave_pix[wave_index])
                        )

        if self.input['strategy']['method'] in ['imagingapphot', 'coronagraphy', 'tacentroid', 'taphot']:
            wave_index = 0

        # the 3D data products
        r['3d'] = {}
        r['3d']['flux'] = self.flux  # model flux cube
        r['3d']['flux_plus_background'] = self.flux_plus_bg  # model flux cube plus background
        if self.input['strategy']['method'] in ['ifuapphot', 'ifunodinscene', 'ifunodoffscene']:  # IFU mode generates 3D cubes.
            r['3d']['reconstructed'] = self.cube_sim
            r['3d']['reconstructed_signal'] = self.cube_signal
            r['3d']['reconstructed_noise'] = self.cube_noise
            r['3d']['reconstructed_saturation'] = self.cube_saturation
            r['3d']['reconstructed_snr'] = self.cube_signal / self.cube_noise

        # the 2D data products
        r['2d'] = {}
        if self.input['strategy']['method'] in ['ifuapphot', 'ifunodinscene', 'ifunodoffscene']:
            # use cube planes for IFUs, though this may be temporary. the actual detector image for an IFU
            # observation is a set of spectra, one for each IFU slice. populate the detector and SNR images from the
            # planes of the reconstructed cubes, but collapse the saturation cube to show the most severe saturation
            # for each spatial pixel.
            r['2d']['detector'] = self.cube_sim[wave_index, :, :]
            r['2d']['snr'] = self.cube_signal[wave_index, :, :] / self.cube_noise[wave_index, :, :]
            r['2d']['saturation'] = np.amax(self.cube_saturation, axis=0)
        else:
            r['2d']['detector'] = self.detector_signal
            r['2d']['snr'] = self.detector_sn
            r['2d']['saturation'] = self.saturation

        # make original, unrotated versions of engine 2D outputs available for slitless mode
        if self.signal.projection_type in ('slitless', 'multiorder'):
            r['2d']['detector_unrotated'] = self.detector_signal_unrot
            r['2d']['snr_unrotated'] = self.detector_sn_unrot
            r['2d']['saturation_unrotated'] = self.saturation_unrot

        # the 1D data products
        r['1d'] = {}

        # these are implicitly included with each entry in self.curves. break them out
        # explicitly as well.
        # NOTE: We do not do any NaN checking here.  This may cause certain values like SNR
        # to be reported as NaN in 1d, but as 0 in scalars.
        r['1d']['wave_pix'] = self.wave_pix
        r['1d']['wave_calc'] = self.wave
        for k in list(self.curves.keys()):
            r['1d'][k] = self.curves[k]

        rw = float(self.wave_pix[wave_index])
        r['scalar']['sn'] = float(sn[wave_index])

        # Most strategies operate on and extract flux, but some (eg TACentroid) use the same machinery, which is
        # in principle completely general, to extract a properly weighted sum of something else (eg, centroid
        # for TACentroid).  These strategies, the "extracted_flux" is the extracted product of interest, and
        # the "extracted_noise" is the standard deviation of the extracted product.

        # Some strategies may not return a flux. TargetAcqCentroid is one example for now. 
        # They are still called "extracted_flux" in the curves, but we rename them here to avoid confusion.
        # The "extracted_noise" is still the standard deviation of the extracted product.
        if self.input['strategy']['method'] in ['tacentroid']:
            r['scalar']['extracted_centroid'] = float(self.curves['extracted_flux'][1][wave_index])
            r['scalar']['extracted_flux'] = np.nan
        else:
            r['scalar']['extracted_flux'] = float(self.curves['extracted_flux'][1][wave_index])
        
        # if the SN is 0.0 (e.g. due to heavy saturation), then we must not know what the variance is so set it to np.nan
        if np.abs(r['scalar']['sn']) > 0.0:
            r['scalar']['extracted_noise'] = float(self.curves['extracted_noise'][1][wave_index])
        else:
            r['scalar']['extracted_noise'] = np.nan
        r['scalar']['background_total'] = float(self.curves['extracted_bg_total'][1][wave_index])
        r['scalar']['background_sky'] = float(self.curves['extracted_bg_only'][1][wave_index])

        # only check for contamination if background level is above some epsilon value. otherwise round-off
        # errors can cause unstable test results.
        if np.abs(r['scalar']['background_total']) > 1.0e-9:
            r['scalar']['contamination'] = (r['scalar']['background_total'] - r['scalar']['background_sky']) / \
                                            r['scalar']['background_total']
        else:
            r['scalar']['contamination'] = 0.0

        r['scalar']['reference_wavelength'] = rw
        r['scalar']['background'] = self.signal.background.bg_spec.sample(rw)
        r['scalar']['background_area'] = self.background_area
        r['scalar']['extraction_area'] = self.extraction_area

        if 'contrast_separation' in self.input['strategy']:
            contrast_separation = self.input['strategy']['contrast_separation']
            r['scalar']['contrast_separation'] = contrast_separation
            r['scalar']['contrast_azimuth'] = self.input['strategy']['contrast_azimuth']
            r['scalar']['contrast'] = np.interp(contrast_separation, self.curves['contrast'][0], self.curves['contrast'][1])

        # return some information that is not a product of a calculation
        r['information'] = {}

        # this is used by the UI to decide how to make the plots. even though IFUs are spectroscopic,
        # they're plotted as 2D spatial images. they could be shown as either single planes from the
        # reconstructed cubes or some combination of multiple planes.
        if self.input['strategy']['method'] in ['ifuapphot', 'ifunodinscene', 'ifunodoffscene']:
            r['information']['calc_type'] = "image"
        else:
            r['information']['calc_type'] = self.projection_type

        r['information']['exposure_specification'] = self.exposure_specification.__dict__

        # Some strategies like IFU and coronagraphy adds dithers. Some are on-source, others are off-source. 
        # This ensures that all on-source dithers are included in the "total_exposure_time", whereas 
        # off-source time is only reported in "all_dithers_time".
        total_exposure_time = r['information']['exposure_specification']['total_exposure_time']
        if 'n_on_source' in self.extracted:
            r['scalar']['total_exposure_time'] = total_exposure_time * self.extracted['n_on_source']
            r['scalar']['all_dithers_time'] = total_exposure_time * self.extracted['n_total']
        else:
            r['scalar']['total_exposure_time'] = total_exposure_time
            r['scalar']['all_dithers_time'] = total_exposure_time

        # Report on measurement, exposure time (time for a single exposure) and saturation time. 
        # The r['scalar']['exposure_time'] value reported back to the client is not used at this point.
        r['scalar']['exposure_time'] = r['information']['exposure_specification']['exposure_time']
        r['scalar']['measurement_time'] = r['information']['exposure_specification']['measurement_time']
        r['scalar']['saturation_time'] = r['information']['exposure_specification']['saturation_time']
        r['scalar']['total_integrations'] = r['information']['exposure_specification']['total_integrations']

        # Report the exposure duty cycle
        duty_cycle = r['information']['exposure_specification']['duty_cycle']
        r['scalar']['duty_cycle'] = duty_cycle

        # get the pixel CR rate in events/s and convert to events/ramp
        r['scalar']['cr_ramp_rate'] = self.noise.pix_cr_rate * r['information']['exposure_specification']['saturation_time']

        # put some instrument and strategy configuration info into the scalar report
        r['scalar']['filter'] = self.input['configuration']['instrument']['filter']
        r['scalar']['disperser'] = self.input['configuration']['instrument']['disperser']

        # target_xy is not found in MSASpecApPhot and may not be found in future strategies as well.
        # if it's not there, assume the target position is the center of the FOV, i.e. (0.0, 0.0).
        if 'target_xy' in self.input['strategy']:
            r['scalar']['x_offset'] = self.input['strategy']['target_xy'][0]
            r['scalar']['y_offset'] = self.input['strategy']['target_xy'][1]
        else:
            r['scalar']['x_offset'] = 0.0
            r['scalar']['y_offset'] = 0.0

        # not all strategies define an aperture size. MSASpecApPhot doesn't because the size is defined by an
        # MSA aperture. optimal extraction strategies likely will not use this, either, once implemented.
        if 'aperture_size' in self.input['strategy']:
            r['scalar']['aperture_size'] = self.input['strategy']['aperture_size']
        else:
            r['scalar']['aperture_size'] = "N/A"

        # return a copy of the input as well
        r['input'] = self.input

        # not yet fully implemented...
        r['warnings'] = self.warnings

        # count total number of saturated pixels in the scene, and warn if necessary
        n_partial_saturated = (self.saturation == 1).sum()
        n_full_saturated = (self.saturation == 2).sum()
        if n_partial_saturated > 0:
            # At some point, the word nonlinear should be changed in the api
            r['warnings']['nonlinear'] = "Partial saturation:\n There are %d saturated pixels at the end " % n_partial_saturated
            r['warnings']['nonlinear'] += "of a ramp. Partial ramps may still be used in some cases."
        if n_full_saturated > 0:
            r['warnings']['saturated'] = "Full saturation:\n There are %d saturated pixels at the end of the " % n_full_saturated
            r['warnings']['saturated'] += "first group. These pixels cannot be recovered."
        
        # Target acquisition mode checks             
        if self.input['configuration']['instrument']['mode'] == 'target_acq':

            # Target acquisitions allow some number of saturated pixels within the extraction aperture. We
            # count both the partial and full saturated pixels.
            number_saturated_pixels = r['1d']['n_full_saturated'][1][0]+r['1d']['n_partial_saturated'][1][0]
            if number_saturated_pixels > self.signal.current_instrument.max_saturated_pixels:
                msg = "<font color=red><b>TA MAY FAIL</b></font>: Number of fully saturated pixels {} in centroid" \
                      " box exceeds the maximum number of {} pixels allowed to ensure a successful " \
                      "target acquisition. Recommend to adjust Detector Setup and/or Instrument " \
                      "Setup accordingly.".format(
                    number_saturated_pixels,
                    self.signal.current_instrument.max_saturated_pixels
                )
                r['warnings'].update({'ta_max_saturated_pixels': msg})

            # Compare to the target acquisition SNR requirement. If target is fully saturated, there is no need to
            # check SNR (since it will be 0)
            if (r['scalar']['sn'] < self.signal.current_instrument.min_snr_threshold) & \
                    (not r['1d']['n_full_saturated'][1][0] > 0):
                msg = "<font color=red><b>TA MAY FAIL</b></font>: calculated SNR {:.2f} must " \
                      "exceed {} to ensure TA success.".format(
                    r['scalar']['sn'], self.signal.current_instrument.min_snr_threshold
                )
                r['warnings'].update({'ta_snr_threshold': msg})
        
        return r

    def as_fits(self):
        """
        This takes the output of self.as_dict() and generates pyfits PrimaryHDU objects
        from the image data.

        Returns
        -------
        output: dict
            dictionary containing pyfits HDUList versions of the image data in report, r.
            The keys match those used in the input.  r['2D']['Grid'] is not carried over,
            though, because the information it contains is incorporated in the FITS headers
            where required.
        """
        r = self.as_dict()

        # add WCS information from the signal that describes the detector plane coordinates.
        # this can be either spatial vs spatial or spatial vs wavelength.
        detector_header = self.pix_grid.wcs_info()

        output = copy.deepcopy(r)

        # handle the sub-reports
        output['sub_reports'] = []
        for report in self.sub_reports:
            output['sub_reports'].append(report.as_fits())

        # first tackle the 3D model cubes, 1 per aperture slice...
        for k in ['flux', 'flux_plus_background']:
            for i in range(len(r['3d'][k])):
                # the cube data coming out of pandeia.engine is ordered x,y,wavelength which is
                # backwards from the numpy convention of ordering axes from slowest to fastest.
                # it is set up this way because we need to broadcast 1D vectors (e.g. througput
                # vs. wavelength) onto the 3D cubes. for this to work, wavelength must be the 3rd
                # axis. so transpose the cube and then flip the Y axis to get the expected
                # orientation in the FITS data as viewed in DS9 or the like.
                o = fits.PrimaryHDU(r['3d'][k][i].transpose()[:, ::-1, :])
                tbhdu, header = self.signal.cube_wcs_info()
                o.header.update(header)
                output['3d'][k][i] = fits.HDUList([o, tbhdu])

        if r['input']['strategy']['method'] in ['ifuapphot', 'ifunodinscene', 'ifunodoffscene']:
            header = self.pix_grid.wcs_info()
            t = self.signal.spectral_detector_transform()
            header['ctype3'] = 'Wavelength'
            header['crpix3'] = 1
            header['crval3'] = t['wave_det_min'] - 0.5 * t['wave_det_step']
            header['cdelt3'] = t['wave_det_step']
            header['cunit3'] = 'um'
            header['cname3'] = 'Wavelength'
            for k in ['', '_signal', '_noise', '_snr', '_saturation']:
                key = 'reconstructed' + k
                # the reconstructed cubes are in the proper z, y, x order...
                o = fits.PrimaryHDU(r['3d'][key][::-1, ::-1, :])
                o.header.update(header)
                output['3d'][key] = o

        # now the 2D data
        for k in r['2d']:
            # flip the Y axis to get the FITS data to look the
            # same in DS9 as in matplotlib.
            o = fits.PrimaryHDU(r['2d'][k][::-1, :])
            o.header.update(detector_header)
            output['2d'][k] = o

        # now the 1D data as binary FITS tables
        for k in list(self.curves.keys()):
            tbhdu = fits.BinTableHDU.from_columns([
                fits.Column(name='WAVELENGTH',
                            unit='um',
                            format="1D",
                            array=self.curves[k][0]),
                fits.Column(name=k,
                            format="1D",
                            array=self.curves[k][1])
            ])
            tbhdu.name = k
            output['1d'][k] = fits.HDUList([tbhdu])
        for k in ['wave_calc', 'wave_pix']:
            tbhdu = fits.BinTableHDU.from_columns([
                fits.Column(name='WAVELENGTH',
                            unit='um',
                            format="1D",
                            array=output['1d'][k])
            ])
            tbhdu.name = k
            output['1d'][k] = fits.HDUList([tbhdu])

        return output
