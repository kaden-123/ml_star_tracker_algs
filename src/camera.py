import numpy as np
import pandas as pd
from attitude import (
    e_to_q,
    q_to_e,
    e_to_DCM,
    DCM_to_e,
    DCM_to_q,
    q_to_DCM,
    q_mul
)

class Camera:
    def __init__(self, data):
        self.data = data
        self.direction = np.array([1.0, 0.0, 0.0, 0.0])
        coords = self.data[["x", "y", "z"]].to_numpy()
        coords /= np.linalg.norm(coords, axis = 1, keepdims=1) 
        self.data = pd.concat([pd.DataFrame(coords, columns = ["ux", "uy", "uz"], index=data.index), data], axis=1)
         
    def celest_point(self, ra, dec, roll):
        """ point cam to given ra, dec, and roll."""
        yaw = ra * 15
        pitch = -dec
        self.direction = e_to_q(np.array([np.deg2rad(roll), np.deg2rad(pitch), np.deg2rad(yaw)]))
        
    def id_point(self, id, roll):
        """ points camera to star given hr id """
        row = self.data.loc[self.data["hr"].eq(id)].iloc[0]
        self.celest_point(row["ra"], row["dec"], roll)
        
    def rand_point(self):
        """ points to random direction (using shoemake alg) """
        u1, u2, u3 = np.random.uniform(0, 1, 3)
        w = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
        x = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
        y = np.sqrt(u1)     * np.sin(2 * np.pi * u3)
        z = np.sqrt(u1)     * np.cos(2 * np.pi * u3)
        self.direction = np.array([w, x, y, z])
        
    def create_centroids(self, focal, res, pixel_pitch, sensor_size):
        """ returns np array [id, px, py] of simulated visible centroids 
            assumes pixel pitch 1 if None 
        """
        # TODO: make option to only put in FOV
        W, H = res
        if sensor_size is not None:
            sx, sy = sensor_size
            #assumes that pixel pitch is same for both horizontal and vertical
            pixel_pitch = sx / W if pixel_pitch is None else pixel_pitch
        else:
            pixel_pitch = 1 if pixel_pitch is None else pixel_pitch
            sx = W * pixel_pitch   # sensor width  in mm
            sy = H * pixel_pitch   # sensor height in mm

        fovx = 2 * np.arctan(sx / (2 * focal))   
        fovy = 2 * np.arctan(sy / (2 * focal))

        #use quaternion as a rotation to turn rotate into camera frame
        celest_to_cam_DCM = (q_to_DCM(self.direction))
        celest_star_direc = np.array(self.data[["ux", "uy", "uz"]])
        cam_star_direc = celest_star_direc @ celest_to_cam_DCM.T
        
        #cam boresight is (1, 0, 0) i think z+ is more standard but should still work the same
        angles_y = np.arctan2(cam_star_direc[:, 1], cam_star_direc[:, 0])
        angles_z = np.arctan2(cam_star_direc[:, 2], cam_star_direc[:, 0])

        mask = (cam_star_direc[:, 0] > 0) & (np.abs(angles_y) <= fovx / 2) & (np.abs(angles_z) <= fovy / 2)

        #reappend to keep ids and then filter out visible stars
        ids = np.array(self.data["hr"])
        centroids = (np.column_stack((ids.reshape(-1,1), cam_star_direc)))
        centroids = centroids[mask]

        #project into image
        f_px = focal / pixel_pitch
        img_px = f_px * (centroids[:, 2] / centroids[:, 1]) + res[0] / 2
        img_py = f_px * (centroids[:, 3] / centroids[:, 1]) + res[1] / 2 
        
        #return new pandas df (maybe just think of returning np directly)
        return pd.DataFrame(np.column_stack([ids[mask], img_px, img_py]), columns=["hr", "px", "py"])

    
    def roll_camera(camera, m):
        """Roll the current camera orientation by n degrees"""
        half_angle = np.deg2rad(m) / 2
        q_roll = np.array([
            np.cos(half_angle), np.sin(half_angle), 0.0, 0.0,
        ])
    
        camera.direction = q_mul(camera.direction, q_roll)
        camera.direction /= np.linalg.norm(camera.direction)
        