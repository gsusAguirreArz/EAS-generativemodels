from cProfile import label
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse


## Load data
def read_npz_file( path: str, info: bool=False ) -> list:
    data = np.load(path)
    labels = {}

    for key in data:
        labels[key] = 0
        if info:
            print(f"Key: '{key}',   Shape: {data[key].shape}")

    return data


def get_data( options: bool=False ) -> list:
    data = read_npz_file("./data/airshower.npz", info=options)
    return [
        data["signal"], 
        data["time"], 
        data["logE"], 
        data["mass"], 
        data["Xmax"], 
        # data['showermax'],
        # data['showeraxis'],
        # data['showercore'],
        # data["detector"]
    ]


## Plot signals
def detectors_grid( n_detectors: int=9 ) -> np.array:
    n0 = (n_detectors-1)/2
    return (np.mgrid[0:n_detectors, 0:n_detectors].astype(np.float32) - n0)


def plot_signals_arrival_times( signals_batch: np.array, labels=[],n_detectors: int=9, N: int=2, random: bool=False, grid: bool=True, show: int=False ) -> matplotlib.figure.Figure:
    fig, axes = plt.subplots(nrows=N, ncols=N, figsize=(13,10), dpi=100)
    axes = axes.squeeze()
    axes = axes.flatten()

    iterator = np.random.choice(signals_batch.shape[0], N*N) if random else np.arange(N*N)

    title = "Energy: {}" if len(labels) != 0 else "Signal"

    for i,j in enumerate(iterator):
        ax = axes[i]
        signal = signals_batch[j].reshape(n_detectors, n_detectors)

        if len(labels) != 0:
            title = title.format(labels[j])

        ## Plot detectors grid
        xd, yd = detectors_grid(n_detectors=n_detectors)
        ax.scatter(xd, yd, c="grey", s=10, alpha=0.3, label="silent")

        ## Plot arrival signal
        mask = signal != 0
        mask[int((n_detectors+1)/2),int((n_detectors+1)/2)] = True
        triggered_detectors = ax.scatter(xd[mask], yd[mask], c=signal[mask], s=100, alpha=1, label="loud") 
        color_bar = fig.colorbar( triggered_detectors, ax=ax)
        color_bar.set_label("arrival time")
        ax.grid(grid)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)
    
    fig.tight_layout()
    if not show:
        plt.close()
    return fig


## Preprocess data
def showerMapDomain( shower: np.array, interval: list=[-1,1] ) -> np.array:
    a,b = interval
    mask = shower != 0
    min_val = shower.min()
    max_val = shower.max()
    shower[mask] = (b-a)*( (shower[mask]-min_val) / (max_val - min_val) ) + a
    return shower


def arrivalTimesDomMap( arrivaltimes: np.array ) -> np.array:
    arrivaltimes[np.isnan(arrivaltimes)] = 0.
    for shower in arrivaltimes:
        shower = showerMapDomain(shower, [1,2])
    return arrivaltimes


def proposedArrivalTimesNorm( arrivaltimes: np.array ) -> np.array:
    arrivaltimes[np.isnan(arrivaltimes)] = 0.
    for shower in arrivaltimes:
        mask = shower != 0
        shower[mask] = shower[mask] - shower.mean()
    arrivaltimes /= arrivaltimes.std()
    arrivaltimes = arrivaltimes.reshape(-1, 1, 9, 9)
    return arrivaltimes


def proposedTotalSignals( detectorsignals: np.array ) -> np.array:
    total_signals = detectorsignals.sum(axis=3)
    total_signals = np.log10(total_signals + 1)
    total_signals[np.isnan(total_signals)] = 0.
    total_signals = total_signals.reshape(-1, 1, 9, 9)
    return total_signals

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-f", "--filepath", required=True)
#     parser.add_argument("-o", "--output", required=False)
#     args = parser.parse_args()
    
#     DATA_FILE_PATH = args.filepath
#     IMG_FILENAME = args.output if args.output != None else "simualtion.jpg"

#     shower_maps, energy = read_npz_file(DATA_FILE_PATH, options=True)
#     plot = plot_signals_arrival_times(shower_maps, N=3, random=True)
#     plot.savefig(IMG_FILENAME, dpi=200)