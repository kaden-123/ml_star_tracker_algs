import random
import numpy as np
import pandas as pd
import h5py
import argparse
from pathlib import Path
from ..camera import Camera

ROOT = Path(__file__).resolve().parent.parent.parent.parent

def make_samples(cam, n_bin, drop_rate, add_rate, 
                 label, n_samples, seed, sigma, 
                 roll_rate, hr_to_idx, res, focal):
    cam.id_point(label, 0)
    center = np.array(res) / 2
    max_radi = np.sqrt((res[0]**2) + (res[1]**2)) / 2 
    bins = np.linspace(0, max_radi, n_bin + 1)

    rng = np.random.default_rng(seed)

    for n in range(n_samples):
        cam.id_point(label, roll_rate * n)

        centroids = cam.create_centroids(focal, res, None, None) 
        centroids = centroids[centroids["hr"].ne(label)]
    
        drop_n_stars = int(centroids["hr"].size * drop_rate)
        add_n_stars = int(centroids["hr"].size * add_rate)
        
        coords = np.array(centroids[["px", "py"]])
        coords += rng.normal(0, sigma, coords.shape)

        visible = ((coords[:, 0] >= 0) & (coords[:, 0] < res[0]) &
                   (coords[:, 1] >= 0) & (coords[:, 1] < res[1]))
        coords = coords[visible] 

        drop_count = min(len(coords), int(round(len(coords) * drop_rate)))
        if drop_count:
            keep = np.ones(len(coords), dtype=bool)
            keep[rng.choice(len(coords), size=drop_count, replace=False)] = False
            coords = coords[keep]

        add_count = int(round(len(coords) * add_rate))
        if add_count:
            false_coords = np.column_stack(
                (
                    rng.uniform(0.0, res[0], size=add_count),
                    rng.uniform(0.0, res[1], size=add_count),
                )
            )
            coords = np.vstack((coords, false_coords))

        distances = np.linalg.norm(coords - center, axis=1)
        histo = np.histogram(distances, bins)
        result = histo[0].astype(np.float32)
        result /= max(result.sum(), 1.0)
        
        data_path = ROOT / "data"
        with h5py.File(data_path, "a") as f:
            mapped_label = hr_to_idx[label]
            if str(mapped_label) not in f:
                grp = f.create_group(str(mapped_label))
            else:
                grp = f[str(mapped_label)]
            
            grp.create_dataset(f"sample_{n}", data=result)

def unit_interval(value):
    number = float(value)
    if not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return number


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return number


def nonnegative_float(value):
    number = float(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be at least 0")
    return number


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate star data")

    parser.add_argument(
        "-f", "--focal", type=positive_int, default=671,
        help="Focal Length of the simulated camera (default: %(default)s (30 degrees FOV))",
    )
    parser.add_argument(
        "-r", "--res", nargs=2, type=positive_int, default=[360, 360],
        help="Resolution of the simulated camera (default: %(default)s)",
    )
    parser.add_argument(
        "-b", "--bins", type=positive_int, default=25,
        help="Number of distance bins (default: %(default)s)",
    )
    parser.add_argument(
        "-d", "--drop-rate", dest="drop_rate", type=unit_interval, default=0.0,
        help="Star drop rate from 0 to 1 (default: %(default)s)",
    )
    parser.add_argument(
        "-a", "--add-rate", dest="add_rate", type=unit_interval, default=0.0,
        help="Star add rate from 0 to 1 (default: %(default)s)",
    )
    parser.add_argument(
        "-n", "--n-samples", dest="n_samples", type=positive_int, default=360,
        help="Number of samples per star (default: %(default)s)",
    )
    parser.add_argument(
        "-s", "--seed", type=int, default=42,
        help="Randomization seed (default: %(default)s)",
    )
    parser.add_argument(
        "-S", "--sigma", type=nonnegative_float, default=0.0,
        help="Sigma spread of generated pixels (default: %(default)s)",
    )
    parser.add_argument(
        "-R", "--roll-rate", dest="roll_rate", type=positive_int, default=1,
        help=(
            "Camera rotation per sample. For example, 360 samples and "
            "a roll rate of 1 means one sample every degree "
            "(default: %(default)s)."
        ),
    )
    return parser.parse_args()


def gen():
    seed = 1
    random.seed(seed)
    print(ROOT)
    data_path = ROOT / "data" / "hygdata_v42.csv"
    if not data_path.exists():
        FileNotFoundError(f"{data_path} is missing. Please download the data from the link in README.md first.")

    #bins, drop_rate, add_rate, n_samples, seed, sigma, roll_rate
    args = parse_arguments()

    data = pd.read_csv(data_path)
    data.drop_duplicates(subset="hr", inplace=True)
    mask = data["hr"].notna()
    cam = Camera(data[mask])

    #some hr ids are missing so indexes are not necessarily correct
    unique_hr_ids = sorted(set(hr for hr in data["hr"]))
    hr_to_idx = {hr_id: i for i, hr_id in enumerate(unique_hr_ids)}
    idx_to_hr = {i: hr_id for hr_id, i in hr_to_idx.items()} 

    s = 0
    for index, row in data[mask].iterrows():
        label = row["hr"]
        print(f"Making data for hr:{label} index: {hr_to_idx[label]}!")
        roll_rate = 1
        make_samples(cam, args.bins, args.drop_rate, args.add_rate, label, 
                    args.n_samples, seed, args.sigma, args.roll_rate, hr_to_idx, args.res, args.focal)    
        s += 1
    print("Done!")
    print(f"{s} Stars accounted for!")

if __name__ == "__main__":
    gen()
