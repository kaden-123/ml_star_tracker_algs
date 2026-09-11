import torch
import numpy as np
from pathlib import Path
from .pole_nn.pole_nn import pole_nn

ROOT = Path(__file__).resolve().parent.parent.parent

def pole_nn_eval(cam, idx_to_hr, FOCAL, RES, CX, CY, coords):
    N_BINS = 25
    N_STAR_CLASSES = 9029
    HIDDEN_ONE = 128
    HIDDEN_TWO = 128
    N_CANDIDATES = 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = pole_nn(N_BINS, N_STAR_CLASSES, HIDDEN_ONE, HIDDEN_TWO)
    model_name = input("Input .pth file to use as model weights: ")
    model_path = ROOT / "data" / model_name
    model.load_state_dict(torch.load(model_path, weights_only=True))

    center = np.array([RES[1] / 2, RES[0] / 2])
    center_distance = np.linalg.norm((coords[["px", "py"]] - center), axis = 1)
    closest_stars = np.argpartition(center_distance, N_CANDIDATES - 1)[:N_CANDIDATES]

    max_radi = np.sqrt((RES[0]**2) + (RES[1]**2)) / 2
    bins = np.linspace(0, max_radi, N_BINS + 1)

    pred_ids = []

    model.eval()

    with torch.inference_mode():
        for candidate_idx in closest_stars:
            guide_star = coords.iloc[candidate_idx][["px", "py"]]
            
            coords_copy = coords[["px", "py"]].copy().to_numpy()
            coords_copy  = np.delete(coords_copy, candidate_idx, axis=0)
            
            guide_vector = (center - guide_star).to_numpy()
            coords_copy += guide_vector
    
            visible = ((coords_copy[:, 0] >= 0) & (coords_copy[:, 0] < RES[0]) &
                       (coords_copy[:, 1] >= 0) & (coords_copy[:, 1] < RES[1]))
            coords_copy = coords_copy[visible]
    
            distances = np.linalg.norm(coords_copy - center, axis=1)
            histo = np.histogram(distances, bins)
            
            result = histo[0].astype(np.float32)
            result /= max(result.sum(), 1.0)
    
            x = torch.from_numpy(result).unsqueeze(0)
            pred_id = model(x).argmax(dim=1).item()
            pred_ids.append(idx_to_hr[pred_id])

    return closest_stars, pred_ids


    

