import os
import numpy as np
import pandas as pd
import pymc3 as pm
from datetime import datetime

import attrici.datahandler as dh
import attrici.const as c
import attrici.models as models
import attrici.fourier as fourier

model_for_var = {
    "tas": models.Tas,
    "tasrange": models.Tasrange,
    "tasskew": models.Tasskew,
    "pr": models.PrecipitationLongtermTrendSigma,
    "hurs": models.Hurs,
    "wind": models.WindLogistic,
    "ps": models.Ps,
    "rsds": models.Rsds,
    "rlds": models.RldsConstSigma,
}


class estimator(object):
    def __init__(self, cfg):

        self.output_dir = cfg.output_dir
        self.draws = cfg.draws
        self.cores = cfg.ncores_per_job
        self.chains = cfg.chains
        self.tune = cfg.tune
        self.subset = cfg.subset
        self.seed = cfg.seed
        self.progressbar = cfg.progressbar
        self.variable = cfg.variable
        self.modes = cfg.modes
        self.f_rescale = c.mask_and_scale[cfg.variable][1]
        self.qm_ref_period = cfg.qm_ref_period
        self.save_trace = cfg.save_trace
        self.report_variables = cfg.report_variables
        self.inference = cfg.inference

        try:
            self.statmodel = model_for_var[self.variable](self.modes)

        except KeyError as error:
            print(
                "No statistical model for this variable. Probably treated as part of other variables."
            )
            raise error

    def estimate_parameters(self, df, lat, lon):

        x_fourier = fourier.get_fourier_valid(df, self.modes)
        x_fourier_01 = (x_fourier + 1) / 2
        x_fourier_01.columns = ["pos" + col for col in x_fourier_01.columns]

        dff = pd.concat([df, x_fourier, x_fourier_01], axis=1)
        df_subset = dh.get_subset(dff, self.subset, self.seed)

        self.model = self.statmodel.setup(df_subset)

        outdir_for_cell = dh.make_cell_output_dir(
            self.output_dir, "traces", lat, lon, self.variable
        )

        # FIXME: Rework loading old traces
        # print("Search for trace in\n", outdir_for_cell)
        # As load_trace does not throw an error when no saved data exists, we here
        # test this manually. FIXME: Could be improved, as we check for existence
        # of names and number of chains only, but not that the data is not corrupted.
        try:
            trace = pm.load_trace(outdir_for_cell, model=self.model)
            print(trace.varnames)
            #     for var in self.statmodel.vars_to_estimate:
            #         if var not in trace.varnames:
            #             print(var, "is not in trace, rerun sampling.")
            #             raise IndexError
            #     if trace.nchains != self.chains:
            #         raise IndexError("Sample data not completely saved. Rerun.")
            print("Successfully loaded sampled data from")
            print(outdir_for_cell)
            print("Skip this for sampling.")
        except Exception as e:
            print("Problem with saved trace:", e, ". Redo parameter estimation.")
            trace = self.sample()
            # print(pm.summary(trace))  # takes too much memory
            if self.save_trace:
                pm.backends.save_trace(trace, outdir_for_cell, overwrite=True)

        return trace, dff

    def sample(self):

        TIME0 = datetime.now()

        if self.inference == "NUTS":
            with self.model:
                trace = pm.sample(
                    draws=self.draws,
                    cores=self.cores,
                    chains=self.chains,
                    tune=self.tune,
                    progressbar=self.progressbar,
                    target_accept=.95
                )
            # could set target_accept=.95 to get smaller step size if warnings appear
        elif self.inference == "ADVI":
            with self.model:
                mean_field = pm.fit(
                    n=10000, method="fullrank_advi", progressbar=self.progressbar
                )
                # TODO: trace is just a workaround here so the rest of the code understands
                # ADVI. We could communicate parameters from mean_fied directly.
                trace = mean_field.sample(1000)
        else:
            raise NotImplementedError

        TIME1 = datetime.now()
        print(
            "Finished job {0} in {1:.0f} seconds.".format(
                os.getpid(), (TIME1 - TIME0).total_seconds()
            )
        )

        return trace

    def estimate_timeseries(self, df, trace, datamin, scale, subtrace=1000):

        # print(trace["mu"].shape, df.shape)
        trace_for_qm = self.statmodel.resample_missing(
            trace, df, subtrace, self.model, self.progressbar
        )

        df_params = dh.create_ref_df(
            df, trace_for_qm, self.qm_ref_period, self.statmodel.params
        )

        cfact_scaled = self.statmodel.quantile_mapping(df_params, df["y_scaled"])
        print("Done with quantile mapping.")

        # fill cfact_scaled as is from quantile mapping
        # for easy checking later
        df.loc[:, "cfact_scaled"] = cfact_scaled

        # rescale all scaled values back to original, invalids included
        df.loc[:, "cfact"] = self.f_rescale(df.loc[:, "cfact_scaled"], datamin, scale)

        # populate invalid values originating from y_scaled with with original values
        invalid_index = df.index[df["y_scaled"].isna()]
        df.loc[invalid_index, "cfact"] = df.loc[invalid_index, "y"]

        # df = df.replace([np.inf, -np.inf], np.nan)
        # if df["y"].isna().sum() > 0:
        yna = df["cfact"].isna()
        yinf = df["cfact"] == np.inf
        yminf = df["cfact"] == -np.inf
        print(f"There are {yna.sum()} NaN values from quantile mapping. Replace.")
        print(f"There are {yinf.sum()} Inf values from quantile mapping. Replace.")
        print(f"There are {yminf.sum()} -Inf values from quantile mapping. Replace.")

        df.loc[yna | yinf | yminf, "cfact"] = df.loc[yna | yinf | yminf, "y"]


        # todo: unifiy indexes so .values can be dropped
        for v in df_params.columns:
            df.loc[:, v] = df_params.loc[:, v].values

        if self.report_variables != "all":
            df = df.loc[:, self.report_variables]

        return df
