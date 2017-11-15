# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Module that implements a background noise calculator (BNC) for the pandeia ETC engine
"""

from __future__ import division, absolute_import

import numpy as np

from .observation import Observation
from .scene import Scene
from .custom_exceptions import EngineInputError
from .instrument_factory import InstrumentFactory
from .strategy import StrategyFactory
from .calc_utils import build_empty_scene
from .etc3D import DetectorSignal, DetectorNoise, CombinedSignal, CalculationConfig


def bnc(calc_input):
    """
    This function uses the pandeia ETC engine to calculate the standard deviation of the per-pixel count rate for a
    given calculation when only the background signal is included. The input can be any valid pandeia calculation.
    The scene information will be ignored and the strategy information ignored except for number of dithers. The output
    will be the maximum standard deviation (sqrt(variance)) encountered at the detector plane.

    Parameters
    ----------
    calc_input: dict
        Pandeia engine input API compliant dict containing information required to perform the background noise calculation

    Returns
    -------
    pix_stddev: float
        Maximum standard deviation in pixel count rate (e-/sec/pixel) generated by the input background signal.
    """

    # ignore the scene in the input calculation and use an empty one instead. the only input signal in the calculation will
    # be the background which is spatially constant across the scene.
    scene_configuration = build_empty_scene()

    try:
        background = calc_input['background']
        instrument_configuration = calc_input['configuration']
        # we need to pass a strategy to Observation and we also need to know the number of dithers to
        # calculate the background variance in the combined detector plane image
        strategy_configuration = calc_input['strategy']
    except KeyError as e:
        message = "Missing information required for the calculation: %s" % str(e)
        raise EngineInputError(value=message)

    # get the calculation configuration from the input or use the defaults
    if 'calculation' in calc_input:
        calc_config = CalculationConfig(config=calc_input['calculation'])
    else:
        calc_config = CalculationConfig()

    scene = Scene(input=scene_configuration)
    instrument = InstrumentFactory(config=instrument_configuration)
    strategy = StrategyFactory(instrument, config=strategy_configuration)

    obs = Observation(
        scene=scene,
        instrument=instrument,
        strategy=strategy,
        background=background
    )

    # seed the random number generator
    seed = obs.get_random_seed()
    np.random.seed(seed=seed)

    # dithering images uncorrelates all of the noise and thus reduces the variance by ndithers:
    #   variance = variance_single/ndithers
    if hasattr(strategy, 'dithers'):
        ndithers = len(strategy.dithers)
    else:
        ndithers = 1

    # Calculate the signal rate in the detector plane. If they're configured, need to loop through
    # configured orders to include all dispersed signal.
    if instrument.projection_type == 'multiorder':
        norders = instrument.disperser_config[instrument.instrument['disperser']]['norders']
        orders = list(range(1, norders + 1))
    else:
        orders = None

    if orders is not None:
        order_signals = []
        for order in orders:
            order_signals.append(DetectorSignal(obs, calc_config=calc_config, order=order))
        signal = CombinedSignal(order_signals)
    else:
        signal = DetectorSignal(obs, calc_config=calc_config)

    noise = DetectorNoise(signal, obs)

    # get the detector plane noise products
    det_var, det_stddev, det_rn_var = noise.on_detector()

    pix_stddev = det_stddev.max() / np.sqrt(ndithers)
    return pix_stddev