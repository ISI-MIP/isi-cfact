import matplotlib.pylab as plt
import cartopy.crs as ccrs
import numpy as np
import netCDF4 as nc
import settings
from plotting.helper_functions import get_path, get_parser

plt.rcParams["figure.figsize"] = 12, 14


def plot_cdo_trend_maps(
        runid, data_dir, variable, dataset, vmax=None, save_fig=False
):
    # data_dir = Path("/p/tmp/mengel/isimip/isi-cfact/output")
    file_dir = data_dir / runid / "cfact" / variable
    ncd = nc.Dataset(file_dir / f"{variable}_{dataset.upper()}_cfactual_trend_2.nc4", "r")

    # Plotting

    # define appropriate colorscheme for the given variable
    if variable == "pr":
        cmap = "BrBG"
        # cmap = 'coolwarm'
    else:
        cmap = "coolwarm"

    # y are the original observations, cfact the counterfactual
    fig = plt.figure()
    figname = f"trend_map.png"
    trend_std = ncd.variables[f"{variable}_orig"][0, ::-1, ::1].std()
    for i, case in enumerate([f"{variable}_orig", variable]):
        data = ncd.variables[case][:]
        # last minus first 30 years
        trend = ncd.variables[case][0, ::-1, ::1]
        if vmax is None:
            vmax = 2
        vmin = -vmax

        ax = plt.subplot(211 + i, projection=ccrs.PlateCarree(central_longitude=0.0))
        ax.coastlines()
        img = ax.imshow(
            trend/trend_std, vmin=vmin, vmax=vmax, extent=[-180, 180, -90, 90], cmap=cmap
        )
        plt.colorbar(img, ax=ax, shrink=0.6)
        ax.grid()
        # ax.plot(loni, lati, "x", markersize=20, markeredgewidth=3, color="r",)
        plt.title(case)

    fig.tight_layout()
    if save_fig:
        figure_dir = data_dir / "figures" / runid / "maps"
        figure_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure_dir / figname, dpi=80)

        print(figure_dir)
    else:
        plt.show()
        return trend, ncd


if __name__ == "__main__":
    parser = get_parser()
    o = parser.parse_args()
    if len(o.runid) > 0:
        for runid in o.runid:
            plot_cdo_trend_maps(
                runid=runid,
                vmax=o.vmax,
                data_dir=settings.output_dir.parents[0],
                variable=settings.variable,
                dataset=settings.dataset.lower(),
                save_fig=True,
            )
    else:
        print("no runid provided")
