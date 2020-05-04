import numpy as np
import pandas as pd
import pymc3 as pm
import attrici.models
import pathlib
import attrici.fourier as fourier
import attrici.datahandler as dh
import settings as s
import pytest
from attrici.models import *
from attrici.distributions import *

cwd = pathlib.Path.cwd()
df = pd.read_hdf(cwd/"tests"/"data"/"ts_GSWP3_lat-20.25_lon-49.75.h5")

x_fourier = fourier.get_fourier_valid(df, s.modes)
concat_notdone = True
if concat_notdone:
    df = pd.concat([df,x_fourier], axis=1)
    concat_notdone = False

df_subset = dh.get_subset(df, s.subset, s.seed)

# todo second parameterization for different modes
# get all subsubclasses of the Distribution class
@pytest.mark.parametrize('Model', [subclass for subclasses in Distribution.__subclasses__() for subclass in subclasses.__subclasses__()])
def test_precip_parameter_bounds(Model):

    # modes are dummy in the longterm case
    model = Model([1, 1, 1, 1])
    model.test = True
    smodel = model.setup(df_subset)

    with smodel:
        trace = pm.sample()

    for param, bounds in model.parameter_bounds.items():

        assert trace[param].min() < trace[param].max()

        if bounds[0] is not None:
            assert trace[param].min() >= bounds[0]
        # todo the other tests are parameter specific and should also be formulated as properties
        #        else:
        #            assert trace[param].min() < -5

        if bounds[1] is not None:
            assert trace[param].max() <= bounds[1]


#        else:
#            assert trace[param].max() > 5
        assert trace[param].min() < trace[param].max(), f'trace of {param} is constant'

#    assert np.abs(trace["a_sigma"].mean()) < 0.01, "mode of a_sigma should be close to zero." class specific
