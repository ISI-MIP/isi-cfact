import settings
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from plotting.helper_functions import get_parser
import attrici.const as c
import scipy.stats as stats
import attrici.estimator as est


def plot_scaled_timeseries(
    runid,
    data_dir,
    variable,
    lat,
    lon,
    window="1a",
    start="01-01-1900",
    end="31-12-2011",
    save_fig=False,
    test_day=None,
    cfact=False,
):
    """
    For mode = dry_days:        Plots timeseries of the observed and counterfactual dry_day freq.
                                From the hd5 timeseries files, as well as the modeled Bernoulli parameter.
    for mode = wet_days:        Plots timeseries of obs. and counterf. precip. amounts on wet days.
    for mode = wet_days_scaled: Plots times. of obs an counterf. of scaled (to standard-deviation 1) precip amounts
                                on wet days as well as the mu (expactition) parameter of the modeled Gamma dist.
    Parameters
    ----------
    runid
    lat
    lon
    mode
    window
    ax

    Returns
    -------

    """
    if lat is None:
        lats_dir = [
            lat_dir
            for lat_dir in (data_dir / runid / "timeseries" / variable).iterdir()
        ]
    else:
        lats_dir = [data_dir / runid / "timeseries" / variable / f"lat_{lat}"]
    for lat_dir in lats_dir:
        if lon is None:
            files = [file for file in lat_dir.iterdir()]
        else:
            files = [
                data_dir
                / runid
                / "timeseries"
                / variable
                / f"lat_{lat}"
                / f"ts_GSWP3_lat{lat}_lon{lon}.h5"
            ]

        for file in files:
            figure_dir = data_dir / "figures" / runid / lat_dir.name
            figure_dir.mkdir(parents=True, exist_ok=True)
            fig = plt.figure()

            df = pd.read_hdf(file)
            df = df.set_index("ds")

            if variable == "pr":
                fig, axs = plt.subplots(4, 1, figsize=(20, 16))
                axs = axs.reshape(-1)
                mode = "dry_days"
                for key in ["y", "cfact"]:
                    df_mean = (
                        (df[key] <= c.threshold[variable][0]).resample(window).mean()
                    )
                    df_mean.plot(alpha=0.5, title=mode, ax=axs[0])
                for key in ["pbern", "pbern_ref"]:
                    df[key].resample(window).mean().plot(alpha=0.5, ax=axs[0])
                mode = "wet_days"
                for key in ["y", "cfact"]:
                    df_mean = (
                        df[key]
                        .loc[df[key] > c.threshold[variable][0]]
                        .resample(window)
                        .mean()
                    )
                    df_mean.plot(alpha=0.5, title=mode, ax=axs[1])
                mode = "wet_days_scaled"
                df_means = (
                    df[["y_scaled", "mu"]]
                    .loc[df["y"] > c.threshold[variable][0]]
                    .resample(window)
                    .mean()
                )
                df_ref_means = (
                    df[["cfact_scaled", "mu_ref"]]
                    .loc[df["cfact"] > c.threshold[variable][0]]
                    .resample(window)
                    .mean()
                )
                pd.concat([df_means, df_ref_means], axis=1).plot(
                    alpha=0.5, title=mode, ax=axs[2]
                )
                mode = "all"
                df.loc[df["y_scaled"].isna(), "y_scaled"] = 0
                df_means = df["y_scaled"].resample(window).mean()
                df_ref_means = df["cfact_scaled"].resample(window).mean()
                pd.concat([df_means, df_ref_means], axis=1).plot(
                    alpha=0.5, title=mode, ax=axs[3]
                )

            else:
                # distinction by distributions
                distribution = est.model_for_var[variable].__bases__[0].__name__
                if distribution == "Rice":
                    # calculate percentiles
                    scale = df["sigma"]
                    scale_ref = df["sigma_ref"]
                    b = df["nu"] / scale
                    b_ref = df["nu_ref"] / scale_ref
                    for pct in [0.05, 0.5, 0.95]:
                        df[f"{int(pct * 100)}pct"] = stats.rice.ppf(
                            pct, b=b, scale=scale
                        )
                        df[f"{int(pct * 100)}pct_ref"] = stats.rice.ppf(
                            pct, b=b_ref, scale=scale_ref
                        )
                elif distribution == "Normal":
                    # calculate percentiles
                    for pct in [0.05, 0.5, 0.95]:
                        df[f"{int(pct * 100)}pct"] = stats.norm.ppf(
                            pct, loc=df["mu"], scale=df["sigma"]
                        )
                        df[f"{int(pct * 100)}pct_ref"] = stats.norm.ppf(
                            pct, loc=df["mu_ref"], scale=df["sigma_ref"]
                        )
                elif distribution == "Beta":
                    # calculate percentiles
                    for pct in [0.05, 0.5, 0.95]:
                        df[f"{int(pct * 100)}pct"] = stats.beta.ppf(
                            pct, df["alpha"], df["beta"]
                        )
                        df[f"{int(pct * 100)}pct_ref"] = stats.beta.ppf(
                            pct, df["alpha_ref"], df["beta_ref"]
                        )
                elif distribution == "Weibull":
                    # calculate percentiles
                    for pct in [0.05, 0.5, 0.95]:
                        df[f"{int(pct * 100)}pct"] = stats.weibull_min.ppf(
                            pct, df["alpha"], scale=df["beta"]
                        )
                        df[f"{int(pct * 100)}pct_ref"] = stats.weibull_min.ppf(
                            pct, df["alpha_ref"], scale=df["beta_ref"]
                        )
                else:
                    raise NotImplemented(
                        f"Plotting for the {distribution}-Distribution is not implemented."
                    )

                df_plot_frame = df[start:end]

                if test_day is None:
                    df_plot_frame = df.resample(window).mean()[start:end]
                else:
                    day, month = test_day.split("-")
                    df_plot_frame = df_plot_frame.loc[
                        (df_plot_frame.index.day == int(day))
                        & (df_plot_frame.index.month == int(month))
                    ]

                if window == "1d":
                    if not cfact:
                        plt.fill_between(
                            df_plot_frame.index,
                            df_plot_frame["5pct"],
                            df_plot_frame["95pct"],
                            alpha=0.2,
                            interpolate=True,
                            color="black",
                        )

                # else:
                # plt.plot(df_plot_frame['5pct'], label=f'5th pct', alpha=0.5, color='gray')
                # plt.plot(df_plot_frame['95pct'], label=f'95th pct', alpha=0.5, color='gray')
                # if cfact:
                #    plt.plot(df_plot_frame['5pct_ref'], label=f'5th pct ref', alpha=0.5, color='gray', linestyle=':')
                #    plt.plot(df_plot_frame['95pct_ref'], label=f'95th pct ref', alpha=0.5, color='gray', linestyle=':')

                plt.plot(
                    df_plot_frame["y_scaled"], label=f"observed {variable}", alpha=0.5
                )
                plt.plot(
                    df_plot_frame["50pct"], label=f"50th pct", alpha=0.9, color="red"
                )
                if cfact:
                    plt.plot(
                        df_plot_frame["cfact_scaled"],
                        label=f"cfact {variable}",
                        alpha=0.5,
                    )
                    plt.plot(
                        df_plot_frame["50pct_ref"],
                        label=f"50th pct ref",
                        alpha=0.9,
                        color="red",
                        linestyle=":",
                    )

            plt.legend()
            fig.suptitle(file.stem)
            if save_fig:
                plt.savefig(figure_dir / f"{file.stem}.png", dpi=80)
            else:
                plt.show()


if __name__ == "__main__":
    parser = get_parser()
    o = parser.parse_args()
    if len(o.runid) > 0:
        for runid in o.runid:
            for mode in ["dry_days"]:
                plot_scaled_timeseries(
                    runid=runid,
                    lat=o.lat,
                    lon=o.lon,
                    window=o.window,
                    start=o.start,
                    end=o.end,
                    data_dir=settings.output_dir.parents[0],
                    variable=settings.variable,
                    save_fig=True,
                    cfact=o.cfact,
                )
    else:
        print("no runid provided")
