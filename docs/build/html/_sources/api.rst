.. automodule:: pygot
API
===
Import GOT as::

   import pygot 

Preprocessing (pp)
------------------
**Dimension reduction** 

.. autosummary::
   :toctree: .

   preprocessing.GS_VAE

Tools (tl)
------------------

Trajectory Inference (traj)
~~~~~~~~~~~~~~~~~~~~~~~


**Velocity model training (awared with time label)**

.. autosummary::
   :toctree: .

   tools.traj.fit_velocity_model
   tools.traj.velocity
   tools.traj.latent_velocity
   tools.traj.latent2gene_velocity

   tools.traj.simulate_trajectory
   
**Source searching and time labeling (most for snapshot data)**

.. autosummary::
   :toctree: .

   tools.traj.fit_velocity_model_without_time
   tools.traj.determine_source_state



Downstream Analysis (analysis)
~~~~~~~~~~~~~~~~~~~~~~~

**Density and pseudotime estimation**



.. autosummary:: 
   :toctree: generated/

   tools.analysis.ProbabilityModel


**Cell fate prediction**

.. autosummary:: 
   :toctree: generated/

   tools.analysis.CellFate
   tools.analysis.TimeSeriesRoadmap


**Gene regulatory network (GRN) inference**

.. autosummary:: 
   :toctree: generated/
   
   tools.analysis.GRN
   tools.analysis.GRNData
   


Plotting (pl)
-----------------
**Velocity visualization**

.. autosummary:: 
   :toctree: .

   plotting.plot_trajectory
   

**Root visualization**

.. autosummary:: 
   :toctree: .
   
   plotting.plot_root_cell


Datasets
---------------
.. autosummary:: 
   :toctree: .

   datasets.synthetic




