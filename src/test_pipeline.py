"""

Functional Pipeline

"""

import numpy as np
import pandas as pd
import torch
import argparse
from pathlib import Path
from camera import Camera
from model_evaluation import pole_nn_eval
from algorithm import davenportq
from attitude import (
    e_to_q,
    q_to_e,
    e_to_DCM,
    DCM_to_e,   
    DCM_to_q,
    q_to_DCM,
    q_mul
)

MODEL_FUNCTIONS = {
    "pole_nn": pole_nn_eval
}


def hr_interval(value):
    number = float(value)
    if not 0 <= number <= 9029:
        raise argparse.ArgumentTypeError("No HR IDS outside of 0 to 9029")
    return number

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model", required=True, 
                                        choices=["pole_nn"],
                                        help="Only one model available: pole__nn")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-i", "--id_point", nargs=2, type=hr_interval, 
                       help="Given int, point camera to the corresponding HR ID with given roll")

    group.add_argument("-r", "--rand_point", action="store_true", 
                       help="point camera to random point in space")

    group.add_argument("-c", "--celest_point", nargs=3, type=float, 
                       help="point camera given celestial coordinates and lastly roll")

    args = parser.parse_args()
    model_name = args.model

    if args.id_point is not None:
        point = ("id", args.id_point)
    elif args.rand_point:
        point = ("rand_point", args.rand_point)
    else:
        point = ("celest_point", args.celest_point)

    return (model_name, point)

def main():    
    data_path = Path("./data/hygdata_v42.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} is missing. Please download the data from the link in README.md first.")

    data = pd.read_csv(data_path)
    data.drop_duplicates(subset="hr", inplace=True)
    data = data[1:]
    mask = data["hr"].notna()

    cam = Camera(data[mask])

    model_name, point = parse_arguments()

    if point[0] == "id":
        star_id, roll = point[1]
        cam.id_point(star_id, roll)
    elif point[0] == "rand_point":
        cam.rand_point()
    else:
        ra, dec, roll = point[1]
        cam.celest_point(ra, dec, roll)

    unique_hr_ids = sorted(set(hr for hr in data["hr"]))
    hr_to_idx = {hr_id: i for i, hr_id in enumerate(unique_hr_ids)}
    idx_to_hr = {i: hr_id for hr_id, i in hr_to_idx.items()} 

    FOCAL = 671
    RES = [360, 360]
    CX = 180
    CY = 180
    coords = cam.create_centroids(FOCAL, RES, None, None)

    predictions = MODEL_FUNCTIONS[model_name](cam, idx_to_hr, FOCAL, RES, 180, 180, coords)

    reference_stars = cam.data.loc[mask & data["hr"].isin(predictions[1])]
    cam_stars = coords[coords["hr"].isin(predictions[1])].copy()

    #reverse calculation to find directional vectors
    cam_stars["px"] = (cam_stars["px"] - CX) / FOCAL # y coordinate
    cam_stars["py"] = (cam_stars["py"] - CY) / FOCAL # z coordinate
    cam_stars = cam_stars.rename(columns={"px" : "y", "py" : "z", })
    cam_stars["x"] = 1.0
    cam_stars.loc[:, ["y", "z", "x"]] /= np.linalg.norm(cam_stars[["y", "z", "x"]].to_numpy(dtype=float), 
                                                       axis = 1, keepdims=True)
    
    stars = pd.merge(reference_stars[["hr", "ux", "uy", "uz"]], cam_stars, on = "hr", how = "outer").copy()

    #separate when other attitude algorithms are added
    ri = stars.loc[:, ["ux", "uy", "uz"]].to_numpy()
    bi = stars.loc[:, ["x", "y", "z"]].to_numpy()
    a = 1
    q_est = davenportq(bi, ri, 1)

    celest_to_cam_DCM = q_to_DCM(q_est)

    boresight_celestial = celest_to_cam_DCM .T @ np.array([1.0, 0.0, 0.0])
    boresight_celestial /= np.linalg.norm(boresight_celestial)
    ux, uy, uz = boresight_celestial

    ra = (np.degrees(np.arctan2(uy, ux)) % 360) / 15
    dec = np.degrees(np.arcsin(np.clip(uz, -1, 1)))

    q_true = cam.direction / np.linalg.norm(cam.direction)
    error_rad = 2 * np.arccos(np.clip(abs(np.dot(q_est, q_true)), 0.0, 1.0))
    error_arcsec = np.degrees(error_rad) * 3600
    print(f"RA: {ra}   |   DEC: {dec}")
    print(f"Arcsecond Error: {error_arcsec}")

if __name__ == "__main__":
    main()