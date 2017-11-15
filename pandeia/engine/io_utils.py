# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
io_utils - commonly used I/O routines for JSON, FITS, and possibly other formats.
"""

from __future__ import division, absolute_import

import os
import json
import errno
import numpy as np
import astropy.io.fits as fits
import pysynphot as psyn

from .custom_exceptions import EngineInputError, DataError
from .constants import pandeia_waveunits, pandeia_fluxunits


class NumPyArangeEncoder(json.JSONEncoder):

    """
    custom encoder to handle numpy arrays that might show up in inputs and outputs:

    http://stackoverflow.com/questions/11561932/why-does-json-dumpslistnp-arange5-fail-while-json-dumpsnp-arange5-tolis

    Parameters
    ----------
    obj: python object
        currently only supports translating np.ndarray and np.float32 objects

    Returns
    -------
    e: json.JSONEncoder object
        JSONEncoder that now understands the custom data types
    """

    def default(self, obj):
        # turn ndarrays into lists
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # or map(int, obj)
        # numpy.float32's started showing up when refactoring refdata FITS files
        # catch them and make them into normal float()'s
        if isinstance(obj, np.float32):
            return float(obj)
        e = json.JSONEncoder.default(self, obj)
        return e

def get_json_parameters(parameters, field, instrument=None, detector=None, aperture=None, subarray=None, readmode=None, instmode=None):
    """
    This was primarly written (or initially written) for the detector parameters
    to read in the structure so values can be dependent on aperture, subarray
    readmode and/or instrument mode.  An example would be:

        "det_pars": {
        "det_type": "h2rg",
        "dark_current": 0.00636,
        "excessp1": 0.0,
        "excessp2": 0.0,
        "ff_electrons": 62500,
        "fullwell": {
			"default": 65000.0,
			"key_format": ["subarray", "readmode"],
			"full__nrsrapid": 30000.0,
			"s200a1__nrs": 22000.0
		},
        "ipc": true,
        "mingroups": 2,
        "pix_size": 18.0,
        "rn": {
			"default": 16.8,
			"key_format": ["aperture"],
			"sw": 17
		},
        "rn_correlation": true
    }

    This could be made more general (e.g., kwargs for the parameters) but for now
    going to leave as-is.

    :param field:
    :param aperture:
    :param subarray:
    :param readmode:
    :param instmode:
    :return:
    """

    if instrument and detector:
        aperture = instrument['aperture']
        subarray = detector['subarray']
        readmode = detector['readmode']
        instmode = instrument['mode']

    if field in parameters and not isinstance(parameters[field], dict):
        # If the field exists and is not a dict then just return the value
        return parameters[field]

    elif field in parameters and isinstance(parameters, dict):
        # If the field exists and it is a dict then we have to check the key_format
        # In this case the format of the parameter will be something like:
        # "rn": {
        #     "default": 16.8,
        #     "key_format": ["aperature"],
        #     "sw": 17
        # },

        # Create the key lookup
        key_lookup = []
        for kf in parameters[field]["key_format"]:
            if kf == "aperture":
                key_lookup.append(aperture)
            elif kf == "subarray":
                key_lookup.append(subarray)
            elif kf == "readmode":
                key_lookup.append(readmode)
            elif kf == "instmode":
                key_lookup.append(instmode)
            else:
                raise DataError("Unknown key format {} listed in {}".format(kf, parameters[field]))

        # Now squish the list to a "__" separated key and see if it exists
        # If all required keys were passed in then create teh string lookup
        # and then if that key is present in the dict return it.
        if not any([x is None for x in key_lookup]):
            key_lookup = '__'.join(key_lookup)

            if key_lookup in parameters[field]:
                return parameters[field][key_lookup]

        # If they key_lookup was not present or some of the required
        # data was not present then just return the default.
        if "default" in parameters[field]:
            return parameters[field]["default"]
        else:
            raise DataError("There is no default value in {}".format(parameters[field]))

    else:
        raise Exception("Bad key/value {} for det_pars {}".format(field, parameters))

def read_json(filename, raise_except=False, **kwargs):
    """
    read in a JSON format file.  return None if the file is not there.

    Parameters
    ----------
    filename: string
        name of the JSON file to read
    except: bool
        if true, raise exception if file is missing. if false, return empty dict.

    Returns
    -------
    d: python object
        data from JSON file decoded into python object
    """
    try:
        with open(filename, 'r') as f:
            json_data = json.load(f, **kwargs)
    except IOError as e:
        if e.errno == errno.ENOENT and raise_except is False:  # No such file
            json_data = {}
        else:
            msg = "Missing JSON file: %s" % filename
            raise EngineInputError(value=msg)
    d = json_data
    return d


def write_json(data, filename, **kwargs):
    """
    write python object into a JSON format file

    Parameters
    ----------
    data: python object
        python object to encode and write to JSON
    filename: string
        name of file to write JSON-encoded data to
    """
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4, cls=NumPyArangeEncoder, separators=(',', ': '), **kwargs)
        f.write('\n')


def append_json(obj, filename):
    """
    read data structure from JSON, append it to a python object, and return the result.
    currently only supported for lists and dicts

    Parameters
    ----------
    obj: python object
        object to be appended to
    filename: string
        filename containing JSON data to be appended

    Returns
    -------
    obj: python object
        input python object now updated with appended data
    """
    json_data = read_json(filename)
    if isinstance(obj, list):
        if isinstance(json_data, list):
            for k in json_data:
                obj.append(k)
        else:
            obj.append(json_data)
    elif isinstance(obj, dict):
        if isinstance(json_data, dict):
            obj.update(json_data)
    else:
        raise ValueError("Can only append to list or dict.")
    return obj


def ref_data_interp(filename, wave, colname=None):
    """
    Read reference data from a FITS file and interpolate it to a provided wavelength array

    Parameters
    ----------
    filename: str
        Filename of reference data
    wave: numpy.ndarray
        Wavelength vector that the reference data will be interpolated onto
    colname: str
        Name of column within the reference file to read and interpolate

    Returns
    -------
    interp_col: numpy.ndarray
        Vector containing reference data interpolated onto wave
    """
    if colname is None:
        raise EngineInputError(value="Must specify name of column to read from reference file.")
    try:
        data = fits.getdata(filename)
    except IOError as e:
        error_msg = "Error reading reference file: " + filename
        raise DataError(value=error_msg)
    if np.any(np.diff(data['wavelength']) < 0):
        indices = np.where(np.diff(data['wavelength']) < 0)[0]
        error_msg = "Wavelengths must be increasing in reference file: %s\n" % (filename)
        error_msg += "Out-of-order indices: %s" % repr(indices)
        raise DataError(value=error_msg)
    try:
        columns = set(k.name.lower() for k in data.columns)
        if colname.lower() not in columns:
            msg = "Column %s not found in %s" % (colname, filename)
            raise DataError(value=msg)
        interp_col = np.interp(wave, data['wavelength'], data[colname])
    except Exception as e:
        error_msg = "Error interpolating reference file: %s : %s" % (filename, type(e))
        raise DataError(value=error_msg)
    return interp_col


def ref_data_column(filename, colname=None, error_msg="Error loading reference file."):
    """
    Read reference data from a FITS file and provide it directly with no interpolation

    Parameters
    ----------
    filename: str
        Filename of reference data
    colname: str
        Name of column within the reference file to read and return
    error_msg: str
        Custom error message to produce in case of an error

    Returns
    -------
    col: numpy.ndarray
        Vector containing requested reference data
    """
    if colname is None:
        raise EngineInputError(value="Must specify name of column to read from reference file.")
    try:
        data = fits.getdata(filename)
    except IOError as e:
        raise DataError(value="I/O " + error_msg)
    try:
        col = data[colname]
    except KeyError as e:
        raise DataError(value="Column not found in reference file: %s" % colname)
    return col


def read_psyn_spectrum(path):
    """
    Read spectrum in FITS or compliant ascii format (wavelength in angstroms, flux in flam).
    See http://etc.stsci.edu/etcstatic/users_guide/1_ref_5_user_spectra.html for details of
    acceptable files for pysynphot.

    Parameters
    ----------
    path: string
        Full pathname to FITS file containing spectrum

    Returns
    -------
    wave, flux:  1D np.ndarray, 1D np.ndarray
        Wavelength and flux vectors in 'pandeia_waveunits' and 'pandeia_fluxunits', respectively
    """
    sp = psyn.FileSpectrum(path)
    sp.convert(pandeia_waveunits)
    sp.convert(pandeia_fluxunits)
    wave, flux = sp.wave, sp.flux
    return wave, flux


def mkdir_p(path):
    """
    Implement 'mkdir -p' functionality with pure python

    Parameters
    ----------
    path: valid path specification
    """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
