import numpy as np
import matplotlib as mlp
import pandas as pd
import random
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy import ndimage

class Camera:
    def __init__(self, fov, resolution, data):
        self.fov = np.deg2rad(fov);
        self.resolution = resolution

        self.direction = np.zeros(3)

        coords = data[["x", "y", "z"]].to_numpy()
        coords /= np.linalg.norm(coords, axis = 1, keepdims=1) 
        self.data = pd.concat([pd.DataFrame(coords, columns = ["ux", "uy", "uz"], index=data.index), data], axis=1)
        
    def point_random(self):
        random.seed(1)
        self.direction = np.random.normal(size=3)
        self.direction /= np.linalg.norm(self.direction)

    def point_to_vec(self, vec):
        self.direction = vec / np.linalg.norm(vec)

    def point(self, catalog, starID):
        mask = self.data[catalog] == starID
        self.direction = np.array(self.data[mask][["ux", "uy", "uz"]]) 
        self.direction = self.direction.reshape(3,)
        
    def prep(self):
        #find dot between cam direction and star unit vector to filter seeable stars
        dots = np.dot(self.data[["ux", "uy", "uz"]].to_numpy(), self.direction)
        mask = (dots >= np.cos(self.fov / 2))
        self.data = self.data[mask].reset_index(drop=True)
        
        #find x and y axis unit vectors for projection
        tmp = np.array([0, 0, 1])
        xc = np.cross(self.direction, tmp)
        xc /= np.linalg.norm(xc)
        yc = np.cross(self.direction, xc)
        yc /= np.linalg.norm(yc)

        focal = (self.resolution[1] / 2) / np.tan(self.fov / 2)

        #find star vec proj onto plane to find pixels
        star_vec = self.data[["ux", "uy", "uz"]].to_numpy()
        x_proj = np.dot(star_vec[:], xc)
        y_proj = np.dot(star_vec[:], yc)
        z_proj = np.dot(star_vec[:], self.direction)
        
        px = (self.resolution[1] / 2) + focal * x_proj / z_proj
        py = (self.resolution[0] / 2) - focal * y_proj / z_proj

        #append to dataframe
        self.data = pd.concat([self.data, pd.DataFrame(px, columns=["px"]), pd.DataFrame(py, columns=["py"])], axis=1)
        
    def draw_img(self):
        #image stuff yup and yes its kinda broken rn
        self.image = np.zeros((self.resolution[0], self.resolution[1]), dtype=np.float32)
        mag = self.data['mag'].to_numpy()
        intensity = 10 ** (-0.4 * mag)
        
        px = self.data['px'].to_numpy().astype(int)
        py = self.data['py'].to_numpy().astype(int)
                
        self.image[py, px] = intensity
        
        self.image = gaussian_filter(self.image, sigma=2)
        
        # self.image += np.random.normal(loc= 0.0001, scale = 0.0001, size = self.image.shape)
        self.image = np.clip(self.image / self.image.max() * 255, 0, 255).astype(np.uint8)
        im = Image.fromarray(self.image)
        im.show()

    def reset_data(self, data):
        coords = data[["x", "y", "z"]].to_numpy()
        coords /= np.linalg.norm(coords, axis = 1, keepdims=1) 
        self.data = pd.concat([pd.DataFrame(coords, columns = ["ux", "uy", "uz"], index=data.index), data], axis=1)

    # def rotate_img(self, degrees):
    #     self.image = ndimage.rotate(self.image, degrees, reshape=True)