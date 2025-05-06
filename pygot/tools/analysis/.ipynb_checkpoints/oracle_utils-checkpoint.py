# -*- coding: utf-8 -*-

import warnings
import numpy as np
import matplotlib.pyplot as plt
from celloracle.applications import Gradient_calculator
from celloracle.applications.development_module import scatter_value_to_grid_value
from celloracle.visualizations.development_module_visualization import (\
                            plot_pseudotime,
                            plot_reference_flow_on_grid,
                            plot_pseudotime_on_grid)
from scipy.stats import norm as normal
from sklearn.neighbors import NearestNeighbors
plt.rcParams["image.cmap"] = "viridis"  


def _clip_inf_value(data):
    """
    This function replace inf with non-inf max value
    """
    max_without_inf = data[data != np.inf].max()
    data[data == np.inf] = max_without_inf

    #print(max_without_inf)

def normalize_gradient(gradient, method="sqrt"):
    """
    Normalize length of 2D vector
    """

    if method == "sqrt":

        size = np.sqrt(np.power(gradient, 2).sum(axis=1))
        size_sq = np.sqrt(size)
        size_sq[size_sq == 0] = 1
        factor = np.repeat(np.expand_dims(size_sq, axis=1), 2, axis=1)

    return gradient / factor

def calculate_p_mass(embedding, smooth=0.5, steps=(40, 40),
                          n_neighbors=100, n_jobs=4, xylim=((None, None), (None, None))):
    """Calculate the velocity using a points on a regular grid and a gaussian kernel

    Note: the function should work also for n-dimensional grid

    Arguments
    ---------
    embedding:

    smooth: float, smooth=0.5
        Higher value correspond to taking in consideration further points
        the standard deviation of the gaussian kernel is smooth * stepsize
    steps: tuple, default
        the number of steps in the grid for each axis
    n_neighbors:
        number of neighbors to use in the calculation, bigger number should not change too much the results..
        ...as soon as smooth is small
        Higher value correspond to slower execution time
    n_jobs:
        number of processes for parallel computing
    xymin:
        ((xmin, xmax), (ymin, ymax))

    Returns
    -------
    total_p_mass: np.ndarray
        density at each point of the grid

    """

    # Prepare the grid
    grs = []
    for dim_i in range(embedding.shape[1]):
        m, M = np.min(embedding[:, dim_i]), np.max(embedding[:, dim_i])

        if xylim[dim_i][0] is not None:
            m = xylim[dim_i][0]
        if xylim[dim_i][1] is not None:
            M = xylim[dim_i][1]

        m = m - 0.025 * np.abs(M - m)
        M = M + 0.025 * np.abs(M - m)
        gr = np.linspace(m, M, steps[dim_i])
        grs.append(gr)

    meshes_tuple = np.meshgrid(*grs)
    gridpoints_coordinates = np.vstack([i.flat for i in meshes_tuple]).T

    nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=n_jobs)
    nn.fit(embedding)
    dists, neighs = nn.kneighbors(gridpoints_coordinates)

    std = np.mean([(g[1] - g[0]) for g in grs])
    # isotropic gaussian kernel
    gaussian_w = normal.pdf(loc=0, scale=smooth * std, x=dists)
    total_p_mass = gaussian_w.sum(1)
    gridpoints_coordinates

    return total_p_mass, gridpoints_coordinates


def normalize_gradient(gradient, method="sqrt"):
    """
    Normalize length of 2D vector
    """

    if method == "sqrt":

        size = np.sqrt(np.power(gradient, 2).sum(axis=1))
        size_sq = np.sqrt(size)
        size_sq[size_sq == 0] = 1
        factor = np.repeat(np.expand_dims(size_sq, axis=1), 2, axis=1)

    return gradient / factor
class VelocityCalculator(Gradient_calculator):
    def __init__(self, oracle_object=None, adata=None, obsm_key=None, pseudotime_key="Pseudotime", velocity_key='velocity_umap', cell_idx_use=None, name=None, gt=None):
        """
        Estimate the direction of differentiation by calculation gradient of pseudotime on the embedding space.
        Please look at web tutorial for example scripts.

        Args:
            adata (anndata): scRNA-seq data in anndata class
            obsm_key (str): Name of dimensional reduction. You can check the list of dimensional reduction data name with "adata.obsm.keys()"
            pseudotime_key (str): Pseudotime data should be stored in adata.obs[pseudotime_key]. Please set the name of pseudotime data in adata.obs
            velocity_key (str): Velocity of each cell in 2d space (e.g. umap). Velocity data should be stored in adata.obsm[velocity_key]. 
            cluster_column_name (str): If you set cluster_column_name and cluster, you can subset cells to calculate gradient.
                Please look at web tutorial for example codes.
            cluster (str): See above.

        """
        self.cell_idx_use = None
        self.n_neighbors = None
        self.min_mass = None
        self.smooth = None
        self.n_grid = None

        if oracle_object is not None:
            self.load_oracle_object(oracle_object=oracle_object,
                                    cell_idx_use=cell_idx_use,
                                    name=name,
                                    pseudotime_key=pseudotime_key,
                                   velocity_key=velocity_key)

        elif adata is not None:
            self.load_adata(adata=adata, obsm_key=obsm_key,
                            pseudotime_key=pseudotime_key,
                            cell_idx_use=cell_idx_use,
                            name=name,
                           velocity_key=velocity_key)

        elif gt is not None:
            self.embedding = gt.embedding.copy()
            self.mass_filter = gt.mass_filter_whole.copy()
            self.mass_filter_whole = gt.mass_filter_whole.copy()
            self.gridpoints_coordinates = gt.gridpoints_coordinates.copy()

            self.n_neighbors = gt.n_neighbors
            self.min_mass = gt.min_mass
            self.smooth = gt.smooth
            self.n_grid = gt.n_grid
            
    def load_adata(self, adata, obsm_key, cell_idx_use=None, name=None, pseudotime_key="Pseudotime", velocity_key="velocity_umap"):
        super().load_adata(adata, obsm_key, cell_idx_use, name, pseudotime_key)
        self.velocity = adata.obsm[velocity_key].copy()


    def load_oracle_object(self, oracle_object, cell_idx_use=None, name=None, pseudotime_key="Pseudotime", velocity_key="velocity_umap"):
        self.load_adata(adata=oracle_object.adata,
                        obsm_key=oracle_object.embedding_name,
                        cell_idx_use=cell_idx_use,
                        name=name,
                        pseudotime_key=pseudotime_key,
                        velocity_key=velocity_key)
        
    def transfer_data_into_grid(self, normalization = "sqrt", scale_factor = "l2_norm_mean", args={}, plot=False,
                               scale=30, s=1, s_grid=30, show_background=True):
        if not args:
            args = {"method": "knn",
                    "n_knn": 30}

        # Prepare input data_new
        if self.cell_idx_use is None:
            embedding = self.embedding
            grid = self.gridpoints_coordinates
            value0 = self.pseudotime
            value1 = self.velocity
        else:
            embedding = self.embedding[self.cell_idx_use, :]
            grid = self.gridpoints_coordinates
            value0 = self.pseudotime[self.cell_idx_use]
            value1 = self.velocity[self.cell_idx_use]

        # Remove inf
        if np.inf in value0:
            # Clip inf
            warnings.warn("Inf value found in the pseudotime data. The inf value is replaced with non-inf max value.", UserWarning)
            _clip_inf_value(data=value0)

        if np.inf in value1:
            # Clip inf
            warnings.warn("Inf value found in the velocity data. The inf value is replaced with non-inf max value.", UserWarning)
            _clip_inf_value(data=value1)

        # Data calculation for each grid point
        self.pseudotime_on_grid = scatter_value_to_grid_value(embedding=embedding,
                                                              grid=grid,
                                                              value=value0,
                                                              **args)
        
        # Data calculation for each grid point
        grid_velocity = scatter_value_to_grid_value(embedding=embedding,
                                                              grid=grid,
                                                              value=value1,
                                                              **args)
        if normalization == "sqrt":
            grid_velocity = normalize_gradient(grid_velocity, method="sqrt")
        
        if scale_factor == "l2_norm_mean":
            # divide gradient by the mean of l2 norm.
            l2_norm = np.linalg.norm(grid_velocity, ord=2, axis=1)
            scale_factor = 1 / l2_norm.mean()

        self.ref_flow = grid_velocity * scale_factor
        
        if plot:
            fig, ax = plt.subplots(1, 3, figsize=[15, 5])

            s = 10
            s_grid = 20
            show_background = True
            ##
            ax_ = ax[0]
            plot_pseudotime(self, ax=ax_, s=s, show_background=show_background)
            ax_.set_title("Pseudotime")


            ####
            ax_ = ax[1]
            plot_pseudotime_on_grid(self, ax=ax_, s=s_grid, show_background=show_background)
            ax_.set_title("Pseudotime on grid")

            ax_ = ax[2]
            plot_reference_flow_on_grid(self, ax=ax_, scale=scale, show_background=show_background, s=s)
            ax_.set_title("Velocity Grid \n(=Development flow)")