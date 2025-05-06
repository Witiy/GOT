from pathlib import Path
from typing import Optional, Union
import scanpy as sc

def synthetic(
    file_path: Optional[
        Union[str, Path]
    ] = "data/Synthetic/tutorial_synthetic_data.h5ad"
):
    """Sythetic data

    Data generated using simulator from `Pratapa et al. (2020) <https://doi.org/10.1038/s41592-019-0690-6>`__.

    This data is generated from a gene regulatory network of 18 genes, which drives cells differentiate linearly. 
    The groundtruth stored in adata.uns['ref_network'] and the groundtruth
    velocity store in adata.layers['velocity_groundtruth']
    The underlying GRN is 

    .. image:: https://raw.githubusercontent.com/Witiy/WitiyImage/img/img/20240904105501.png
       :width: 200px

    Returns
    -------
    Returns `adata` object
    """
    url = "https://figshare.com/ndownloader/files/48994252"
    adata = sc.read(file_path, backup_url=url, sparse=False, cache=True)
    adata.var_names_make_unique()
    return adata


