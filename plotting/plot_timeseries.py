import settings
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from plotting.helper_functions import get_parser
import netCDF4 as nc


def plot_timeseries(
        runid,
        data_dir,
        variable,
        lat,
        lon,
        input_dir,
        dataset,
        source_file,
        window="1a",
        start="01-01-1900",
        end="31-12-2011",
        save_fig=False,
):
    # get the dates (ds)
    input_file = input_dir / dataset / source_file.lower()
    obs_data = nc.Dataset(input_file, "r")
    nct = obs_data.variables["time"]
    ds = pd.to_datetime(nct[:], unit="D", origin=pd.Timestamp(nct.units.lstrip("days since")))

    lat_dir = data_dir / runid / "timeseries" / variable / f"lat_{lat}"
    file = data_dir / runid / "timeseries" / variable / f"lat_{lat}" / f"ts_{dataset}_lat{lat}_lon{lon}.h5"
    df = pd.read_hdf(file)
    # set index to be dates
    df["ds"] = ds
    df = df.set_index("ds")

    fig = plt.figure()

    df_plot_frame = df.resample(window).mean()[start:end]

    plt.plot(
        df_plot_frame["y"],
        label=f"observed {variable}",
        alpha=0.5
    )

    plt.plot(
        df_plot_frame["cfact"],
        label=f"cfact {variable}",
        alpha=0.5,
    )

    plt.legend()
    fig.suptitle(file.stem)
    if save_fig:
        figure_dir = data_dir / "figures" / runid / lat_dir.name
        figure_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(figure_dir / f"{file.stem}.png", dpi=80)
    else:
        plt.show()


if __name__ == "__main__":
    parser = get_parser()
    o = parser.parse_args()
    if len(o.runid) > 0:
        for runid in o.runid:
            for mode in ["dry_days"]:
                plot_timeseries(
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
