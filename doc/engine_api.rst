Description of ETC API
======================

Installation instructions:
--------------------------
1)   pip install pandeia.engine
2)   download data package from http://ssb.stsci.edu/pandeia/engine/1.X and untar it. 
3)   set the pandeia_refdata environment variable to point to the data, e.g.: export pandeia_refdata=/path/to/pandeia_data

See also https://jwst.stsci.edu/science-planning/proposal-planning-toolbox/exposure-time-calculator-etc for more information.


Basic usage
-----------

A simple dict-based API has been implemented as an interface to the JWST ETC engine.
The use of dicts allows for easy serialization of the data going into and out of the engine.
Here is an example of its usage::

  from pandeia.engine.calc_utils import build_default_calc, build_default_source
  from pandeia.engine.perform_calculation import perform_calculation
  
  # make a default calculation
  input = build_default_calc(telescope='jwst', instrument='nircam', mode='sw_imaging', method='imagingapphot')
  
  # Edit input dict according to the api (see engine_input_api.rst)
  ...
  
  output = perform_calculation(input)

"input" is a dict described by :doc: "engine_input_api.rst" and "output" is described
by :doc: "engine_output_api.rst".

There is also a "fits_report" function that can be imported from "perform_calculation".
It takes the output from "perform_calculation()" and converts the 2D and 3D images to
"astropy.io.fits.PrimaryHDU" format.

Alternatively use:

   output = perform_calculation(input, dict_report=False)
   output_as_fits = output.as_fits()
	
And write elements of the output to a fits file:
   
   output_as_fits['2d']['snr'].writeto('snr_image.fits')

