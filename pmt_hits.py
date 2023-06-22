import numpy as np
import math
import json
from tqdm import tqdm

class PhotonPropagation:
    def __init__(self, x0, y0, n_fotons, arr_times):
        self.x0 = x0
        self.y0 = y0
        self.n_fotons = n_fotons
        self.arr_times = arr_times
        self.z0 = 0
        self.params = self.read_params()
    
    def read_params(self):
        with open('simulation_params.json', 'r') as file:
            params = json.load(file)
        return params

    def random_three_vector(self):
        """
        Generates a random 3D unit vector (direction) with a uniform spherical distribution
        Algo from http://stackoverflow.com/questions/5408276/python-uniform-spherical-distribution
        :return:
        """
        phi = np.random.uniform(0,np.pi*2)
        costheta = np.random.uniform(-1,1)
        theta = np.arccos(costheta)
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        return x ,y, z

    def randomvector(self, n):
        components = [np.random.normal() for i in range(n)]
        r = math.sqrt(sum(x*x for x in components))
        v = [x/r for x in components]
        return v

    def sim_pmt_hits(self, x_0, y_0, fotons):
        hits = {key: 0 for key in ['pmt_1', 'pmt_2', 'pmt_3', 'pmt_4']}
        pmt_positions = self.params['pmt_positions']
        pmt_radius_sq = self.params['pmt_radius']**2
        
        for _ in range(fotons):
            u = self.randomvector(3)
            if u[1] < 0:
                continue
            
            t = (self.params['dist_gem_pmt'] - self.z0) / u[2]
            x = x_0 + t * u[0]
            y = y_0 + t * u[1]
            
            for pmt_name, pmt_pos in pmt_positions.items():
                dx = x - pmt_pos['x']
                dy = y - pmt_pos['y']
                if dx**2 + dy**2 < pmt_radius_sq:
                    hits[pmt_name] += 1
                    break

        return hits

    def pmt_hits(self):
        hits = {}
        for i, (x, y, f, t) in enumerate(zip(self.x0, self.y0, self.n_fotons, self.arr_times)):
            cluster_name = 'cluster_{}'.format(i)
            hits[cluster_name] = self.sim_pmt_hits(x, y, f)
            hits[cluster_name]['arrival_time'] = t
                                                            
        return hits
