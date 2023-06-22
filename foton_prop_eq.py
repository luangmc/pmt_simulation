import numpy as np
import json


class CalculatePmtHits():
    def __init__(self, x0, y0, n_fotons, arr_times):
        self.x0 = x0
        self.y0 = y0
        self.z0 = 0
        self.n_fotons = n_fotons
        self.arr_times = arr_times
        self.params = self.read_params()

    def read_params(self):
        with open('simulation_params.json', 'r') as file:
            params = json.load(file)
        return params

    def calculate_hits(self, pmt_position, x0, y0, n_fotons):
        x_pmt, y_pmt, z_pmt = pmt_position['x'], pmt_position['y'], self.params['dist_gem_pmt']
        r_pmt = self.params['pmt_radius']
        n = 3.9829589
        R = np.sqrt((x_pmt - x0) ** 2 + (y_pmt - y0) ** 2 + (z_pmt - self.z0) ** 2)
        return round(n_fotons * (r_pmt ** 2) * (z_pmt ** 2) / (4 * (R ** n)), 0)

    def pmt_hits(self):
        hits = {}
        pmts_list = ['pmt_1', 'pmt_2', 'pmt_3', 'pmt_4']
        for i, (x, y, f, t) in enumerate(zip(self.x0, self.y0, self.n_fotons, self.arr_times)):
            cluster_name = 'voxel_{}'.format(i)
            hits[cluster_name] = {key: 0 for key in pmts_list}
            for p in pmts_list:
                pmt_position = self.params["pmt_positions"][p]
                hits[cluster_name][p] = int(self.calculate_hits(pmt_position, x, y, f))
            hits[cluster_name]['arrival_time'] = t
        return hits
    
