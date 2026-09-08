import numpy as np
import pandas as pd

def identify_stars(coords, res, model, n_bin, idx_to_hr, n_candidates):
    center = np.array([res[1] / 2, res[0] / 2])
    center_distance = np.linalg.norm((coords[["px", "py"]] - center), axis = 1)
    closest_stars = np.argpartition(center_distance, n_candidates - 1)[:n_candidates]

    max_radi = np.sqrt((res[0]**2) + (res[1]**2)) / 2
    bins = np.linspace(0, max_radi, n_bin + 1)

    pred_ids = []

    model.eval()

    with torch.inference_mode():
        for candidate_idx in closest_stars:
            guide_star = coords.iloc[candidate_idx][["px", "py"]]
            
            coords_copy = coords[["px", "py"]].copy().to_numpy()
            coords_copy  = np.delete(coords_copy, candidate_idx, axis=0)
            
            guide_vector = (center - guide_star).to_numpy()
            coords_copy += guide_vector
    
            visible = ((coords_copy[:, 0] >= 0) & (coords_copy[:, 0] < res[0]) &
                       (coords_copy[:, 1] >= 0) & (coords_copy[:, 1] < res[1]))
            coords_copy = coords_copy[visible]
    
            distances = np.linalg.norm(coords_copy - center, axis=1)
            histo = np.histogram(distances, bins)
            
            result = histo[0].astype(np.float32)
            result /= max(result.sum(), 1.0)
    
            x = torch.from_numpy(result).unsqueeze(0)
            pred_id = model(x).argmax(dim=1).item()
            pred_ids.append(idx_to_hr[pred_id])

    return closest_stars, pred_ids

#takes in array of cam vectors and cat vectors to given optimal quaternion
def davenportq(bi, ri, a):
    B = a * np.matmul(bi, ri.T)
    S = B + B.T
    z = np.array([B[1][2] - B[2][1], B[2][0] - B[0][2], B[0][1] - B[1][0]])
    S - np.identity(3) * np.trace(B)
    z = z.reshape(3, 1)
    K = np.block([
        [np.array([np.trace(B)]), z.T],
        [z, S - np.trace(B) * np.identity(3)]
    ])
    eigens = np.linalg.eig(K)
    q = eigens[1][np.argmax(eigens[0])]
    q = q / np.linalg.norm(q)
    return q


