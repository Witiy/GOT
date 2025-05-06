import numpy as np
cimport numpy as np
cimport cython

cdef class Interpolator:
    cdef np.ndarray dist_map
    cdef np.ndarray X0
    cdef np.ndarray X1
    cdef int l0
    cdef np.float_t dt
    
    def __init__(self, 
                 np.ndarray dist_map,
                 np.ndarray X0,
                 np.ndarray X1,
                 np.float_t dt):
        self.dist_map = dist_map
        self.X0 = X0
        self.X1 = X1
        self.l0 = len(X0)
        self.dt = dt
        
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def mini_batch_interpolate(
        self,
        list[object] shortest_paths_indices ,
        np.ndarray[np.float_t, ndim=1] ts):
        cdef list[np.ndarray] xa_t, va_t
        xa_t, va_t = [], []
        for i in range(len(ts)):
            xa, va = self.interpolate_one_point_shortest_path(shortest_paths_indices[i], ts[i])
            xa_t.append(xa)
            va_t.append(va)
        return xa_t, va_t
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def interpolate_one_point_shortest_path(
        self,
        np.ndarray[np.int_t, ndim=1] shortest_path_indices,
        np.float_t ti=0.5):
        
        cdef int s_idx
        cdef np.float_t total_dist, td, remain_t, norm_va
        cdef np.ndarray[np.float_t , ndim=1] ori_path_dist, path_dist, va
        cdef np.ndarray[np.float_t , ndim=1] start_feature, end_feature
        cdef np.ndarray[np.float_t , ndim=1] xa

        interpolated_features = []
        ori_path_dist = np.array([self.dist_map[shortest_path_indices[0], shortest_path_indices[i]] for i in range(0, len(shortest_path_indices))])
        total_dist = ori_path_dist[len(ori_path_dist) - 1]
        path_dist = ori_path_dist / total_dist
        s_idx = np.argmin(np.abs(path_dist - ti))
        td = path_dist[s_idx]
        remain_t = ti - td

        # Helper function to get features
        def get_features(idx):
            if idx >= self.l0:
                return self.X1[idx - self.l0]
            else:
                return self.X0[idx]
        
        if remain_t > 0:
            start_feature = get_features(shortest_path_indices[s_idx])
            end_feature = get_features(shortest_path_indices[s_idx + 1])
        else:
            start_feature = get_features(shortest_path_indices[s_idx - 1])
            end_feature = get_features(shortest_path_indices[s_idx])
        
        va = end_feature - start_feature
        norm_va = np.linalg.norm(va)
        xa = start_feature + (total_dist / norm_va) * remain_t * va if remain_t > 0 else end_feature + (total_dist / norm_va) * remain_t * va

        va = ((total_dist / norm_va) * self.dt) * va

        return xa, va