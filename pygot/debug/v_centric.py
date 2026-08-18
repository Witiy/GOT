from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import torch

from ..preprocessing import learn_embed2vis_map, load_map_model
from ..tools.traj.v_centric_training import GraphicalOTVelocitySampler


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


@contextmanager
def _temporary_random_seed(random_state):
    if random_state is None:
        yield
        return

    numpy_state = np.random.get_state()
    python_state = random.getstate()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    np.random.seed(random_state)
    random.seed(random_state)
    torch.manual_seed(random_state)
    try:
        yield
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


@dataclass
class VCentricSampleBatch:
    """One directly sampled v-centric training batch and its debug metadata."""

    t: np.ndarray
    x: np.ndarray
    u: np.ndarray
    x0: list
    x1: list
    metadata: dict

    def __len__(self):
        return len(self.x)

    def __getattr__(self, name):
        metadata = self.__dict__.get('metadata', {})
        if name in metadata:
            return metadata[name]
        raise AttributeError(name)


class VCentricSamplingDebugger:
    """Run and visualize the same sampling procedure used by v-centric training."""

    def __init__(
            self,
            adata,
            time_key,
            embedding_key,
            graph_key=None,
            knn_constraint=True,
            n_neighbors=50,
            landmarks=False,
            linear=False,
            path='',
            device=None,
            **sampler_kwargs,
            ):
        if device is None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.adata = adata
        self.time_key = time_key
        self.embedding_key = embedding_key
        self.graph_key = embedding_key if graph_key is None else graph_key
        self.device = device
        self.sampler = GraphicalOTVelocitySampler(
            adata,
            time_key,
            graph_key=self.graph_key,
            embedding_key=embedding_key,
            device=device,
            landmarks=landmarks,
            path=path,
            linear=linear,
            n_neighbors=n_neighbors,
            knn_constraint=knn_constraint,
            **sampler_kwargs,
        )
        self.vis_key = None
        self.vis_mapper = None
        self.mapping_rmse = None

    def sample(
            self,
            batch_size=256,
            n_samples_in_path=1,
            distance_metrics='SP',
            filtered=True,
            add_noise=True,
            sigma=0.1,
            k=15,
            q=80,
            random_state=None,
            ):
        """Sample without constructing or training a velocity model."""
        use_filtered = filtered and self.sampler.knn_constraint
        sample_method = (
            self.sampler.filtered_sample_batch_path
            if use_filtered
            else self.sampler.sample_batch_path
        )
        kwargs = {
            'sigma': sigma,
            'batch_size': batch_size,
            'distance_metrics': distance_metrics,
            'add_noise': add_noise,
            'n_samples_in_path': n_samples_in_path,
            'return_metadata': True,
        }
        if use_filtered:
            kwargs.update({'k': k, 'q': q})

        with _temporary_random_seed(random_state):
            result = sample_method(**kwargs)

        t, x, u, x0, x1, metadata = result
        metadata = self._enrich_metadata(metadata)
        return VCentricSampleBatch(
            t=_to_numpy(t),
            x=_to_numpy(x),
            u=_to_numpy(u),
            x0=[_to_numpy(item) for item in x0],
            x1=[_to_numpy(item) for item in x1],
            metadata=metadata,
        )

    def _enrich_metadata(self, metadata):
        metadata = metadata.copy()
        pair_transition = metadata['pair_transition_idx']
        source_names = []
        target_names = []
        pair_time_start = []
        pair_time_end = []
        for transition, source_idx, target_idx in zip(
                pair_transition,
                metadata['pair_source_idx'],
                metadata['pair_target_idx'],
                ):
            source_names.append(self.sampler.index_list[transition][source_idx])
            target_names.append(self.sampler.index_list[transition + 1][target_idx])
            pair_time_start.append(self.sampler.ts[transition])
            pair_time_end.append(self.sampler.ts[transition + 1])

        metadata['pair_source_obs_names'] = np.asarray(source_names)
        metadata['pair_target_obs_names'] = np.asarray(target_names)
        metadata['pair_time_start'] = np.asarray(pair_time_start)
        metadata['pair_time_end'] = np.asarray(pair_time_end)

        sample_transition = metadata['transition_idx']
        time_start = self.sampler.ts[sample_transition]
        time_end = self.sampler.ts[sample_transition + 1]
        metadata['time_start'] = time_start
        metadata['time_end'] = time_end
        if np.issubdtype(np.asarray(self.sampler.ts).dtype, np.number):
            metadata['data_time'] = (
                (1 - metadata['alpha']) * time_start
                + metadata['alpha'] * time_end
            )
        else:
            metadata['data_time'] = None
        metadata['source_idx'] = metadata['pair_source_idx'][metadata['pair_id']]
        metadata['target_idx'] = metadata['pair_target_idx'][metadata['pair_id']]
        metadata['source_obs_names'] = metadata['pair_source_obs_names'][metadata['pair_id']]
        metadata['target_obs_names'] = metadata['pair_target_obs_names'][metadata['pair_id']]
        return metadata

    def summary(self, batch, print_summary=True):
        """Return basic sampling, connectivity, filtering, and velocity diagnostics."""
        path_length = batch.metadata['pair_path_length']
        chord_length = batch.metadata['pair_chord_length']
        valid_chord = chord_length > 0
        ratios = np.full_like(path_length, np.nan, dtype=float)
        ratios[valid_chord] = path_length[valid_chord] / chord_length[valid_chord]
        velocity_norm = np.linalg.norm(batch.u, axis=1)
        finite_ratios = ratios[np.isfinite(ratios)]
        result = {
            'ot_pairs': len(path_length),
            'connected_pairs': int(batch.metadata['pair_connected'].sum()),
            'disconnected_pairs': int((~batch.metadata['pair_connected']).sum()),
            'linear_fallback_pairs': int(batch.metadata['pair_used_linear_fallback'].sum()),
            'samples_before_filter': len(batch.metadata['pre_filter_x']),
            'samples_after_filter': len(batch.x),
            'mean_velocity_norm': float(np.mean(velocity_norm)),
            'p95_velocity_norm': float(np.percentile(velocity_norm, 95)),
            'max_velocity_norm': float(np.max(velocity_norm)),
            'mean_path_chord_ratio': (
                float(np.mean(finite_ratios)) if len(finite_ratios) else np.nan
            ),
            'max_path_chord_ratio': (
                float(np.max(finite_ratios)) if len(finite_ratios) else np.nan
            ),
        }
        if self.mapping_rmse is not None:
            result['embed_to_vis_rmse'] = self.mapping_rmse
        if print_summary:
            for key, value in result.items():
                print(f'{key}: {value}')
        return result

    def learn_vis_map(
            self,
            vis_key='X_umap',
            batch_size=256,
            num_epochs=100,
            patience=5,
            device=None,
            random_state=None,
            ):
        """Learn an out-of-sample map from the training embedding to a 2D view."""
        if vis_key not in self.adata.obsm:
            raise KeyError(f'`{vis_key}` is not present in `adata.obsm`')
        if self.adata.obsm[vis_key].shape[1] != 2:
            raise ValueError(f'`adata.obsm[{vis_key!r}]` must be two-dimensional')
        with _temporary_random_seed(random_state):
            mapped_cells, mapper = learn_embed2vis_map(
                self.adata,
                embedding_key=self.embedding_key,
                vis_key=vis_key,
                batch_size=batch_size,
                num_epochs=num_epochs,
                patience=patience,
                device=device,
            )
        self.vis_key = vis_key
        self.vis_mapper = mapper
        target = np.asarray(self.adata.obsm[vis_key])
        self.mapping_rmse = float(np.sqrt(np.mean((mapped_cells - target) ** 2)))
        return mapped_cells, mapper

    def set_vis_mapper(self, mapper, vis_key='X_umap'):
        """Reuse an existing embed-to-visualization mapper."""
        if vis_key not in self.adata.obsm:
            raise KeyError(f'`{vis_key}` is not present in `adata.obsm`')
        if self.adata.obsm[vis_key].shape[1] != 2:
            raise ValueError(f'`adata.obsm[{vis_key!r}]` must be two-dimensional')
        mapper.eval()
        self.vis_key = vis_key
        self.vis_mapper = mapper
        mapped = self._map_positions(np.asarray(self.adata.obsm[self.embedding_key]))
        target = np.asarray(self.adata.obsm[vis_key])
        self.mapping_rmse = float(np.sqrt(np.mean((mapped - target) ** 2)))
        return self

    def save_vis_mapper(self, path):
        if self.vis_mapper is None:
            raise RuntimeError('No visualization mapper has been fitted or assigned')
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.vis_mapper.save(str(path))

    def load_vis_mapper(self, path, vis_key='X_umap'):
        return self.set_vis_mapper(load_map_model(str(path)), vis_key=vis_key)

    def _map_positions(self, x):
        x = np.asarray(x, dtype=np.float32)
        if self.vis_mapper is None:
            raise RuntimeError('Call `learn_vis_map` or `set_vis_mapper` first')
        mapped = self.vis_mapper.transform(x)
        if mapped.shape[1] != 2:
            raise ValueError('The visualization mapper output must be two-dimensional')
        return mapped

    def project_to_2d(self, batch, dt=1e-3):
        """Project sample positions and velocities through the learned nonlinear map."""
        if dt <= 0:
            raise ValueError('`dt` must be positive')
        x_vis = self._map_positions(batch.x)
        x_next_vis = self._map_positions(batch.x + dt * batch.u)
        batch.metadata['x_vis'] = x_vis
        batch.metadata['u_vis'] = (x_next_vis - x_vis) / dt
        batch.metadata['pair_x0_vis'] = self._map_positions(batch.metadata['pair_x0'])
        batch.metadata['pair_x1_vis'] = self._map_positions(batch.metadata['pair_x1'])
        batch.metadata['pre_filter_x_vis'] = self._map_positions(batch.metadata['pre_filter_x'])
        return batch

    def plot_2d(
            self,
            batch,
            time_point,
            vis_key=None,
            ax=None,
            learn_map_kwargs=None,
            max_paths=100,
            embedding_kw=None,
            path_kw=None,
            sample_kw=None,
            source_kw=None,
            target_kw=None,
            ):
        """Plot sampled paths and points for one adjacent time transition.

        ``time_point`` is the source label of the transition to plot. For
        example, if the observed time labels are ``[0, 1, 2]``, passing
        ``time_point=1`` plots samples and paths from the ``1 -> 2``
        transition.

        When the fitted embedding is already two-dimensional, its coordinates
        are used directly and ``vis_key`` is unnecessary. For a
        higher-dimensional fitted embedding, ``vis_key`` must identify a 2D
        representation in ``adata.obsm``; an out-of-sample map is learned so
        sampled paths and points can be projected into that representation.
        """
        import matplotlib.pyplot as plt

        transition = self._transition_index(time_point)
        sample_mask = batch.transition_idx == transition
        if not np.any(sample_mask):
            raise ValueError(
                f'No retained samples are available for time point {time_point!r}'
            )

        embedding = np.asarray(self.adata.obsm[self.embedding_key])
        direct_2d = embedding.shape[1] == 2
        if direct_2d:
            observed = embedding
            map_positions = lambda x: np.asarray(x)
            axis_key = self.embedding_key
        else:
            if vis_key is None:
                raise ValueError(
                    '`vis_key` is required when the fitted embedding is not 2D'
                )
            if self.vis_mapper is None or self.vis_key != vis_key:
                self.learn_vis_map(vis_key=vis_key, **(learn_map_kwargs or {}))
            observed = np.asarray(self.adata.obsm[vis_key])
            map_positions = self._map_positions
            axis_key = vis_key

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 6))
        background_style = {'s': 6, 'c': 'lightgray', 'alpha': .45}
        background_style.update(embedding_kw or {})
        ax.scatter(observed[:, 0], observed[:, 1], **background_style)

        pair_ids = np.unique(batch.pair_id[sample_mask])[:max_paths]
        path_style = {'color': 'black', 'alpha': .3, 'linewidth': 1}
        path_style.update(path_kw or {})
        for pair_id in pair_ids:
            path_x = self._pair_path_coordinates(batch, pair_id)
            if path_x is None:
                continue
            path_vis = map_positions(path_x)
            ax.plot(path_vis[:, 0], path_vis[:, 1], **path_style)

        x_vis = map_positions(batch.x[sample_mask])
        sample_style = {'color': 'black', 's': 18, 'alpha': .8, 'zorder': 3}
        sample_style.update(sample_kw or {})
        ax.scatter(x_vis[:, 0], x_vis[:, 1], **sample_style)

        source_vis = map_positions(batch.pair_x0[pair_ids])
        target_vis = map_positions(batch.pair_x1[pair_ids])
        source_style = {'color': 'blue', 's': 20, 'label': 'source', 'zorder': 4}
        source_style.update(source_kw or {})
        ax.scatter(source_vis[:, 0], source_vis[:, 1], **source_style)
        target_style = {
            'facecolors': 'none',
            'edgecolors': 'red',
            's': 20,
            'label': 'target',
            'zorder': 4,
        }
        target_style.update(target_kw or {})
        ax.scatter(target_vis[:, 0], target_vis[:, 1], **target_style)

        time_end = self.sampler.ts[transition + 1]
        ax.set_title(f'V-centric samples: {time_point} → {time_end}')
        ax.set_xlabel(f'{axis_key} 1')
        ax.set_ylabel(f'{axis_key} 2')
        ax.axis('off')
        return ax.figure, ax

    def _transition_index(self, time_point):
        matches = np.flatnonzero(self.sampler.ts[:-1] == time_point)
        if len(matches) == 0:
            available = self.sampler.ts[:-1].tolist()
            raise ValueError(
                f'Unknown source time point {time_point!r}; choose one of {available}'
            )
        return int(matches[0])

    def plot_pair_2d(
            self,
            batch,
            pair_id,
            vis_key='X_umap',
            ax=None,
            show_velocity=True,
            normalize_velocity=True,
            **kwargs,
            ):
        """Plot one OT pair and all samples drawn from its path."""
        import matplotlib.pyplot as plt

        if pair_id < 0 or pair_id >= len(batch.metadata['paths']):
            raise IndexError(f'Unknown pair_id: {pair_id}')
        if self.vis_mapper is None or self.vis_key != vis_key:
            self.learn_vis_map(vis_key=vis_key, **kwargs.pop('learn_map_kwargs', {}))
        self.project_to_2d(batch, dt=kwargs.pop('dt', 1e-3))
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 6))

        observed = np.asarray(self.adata.obsm[vis_key])
        ax.scatter(observed[:, 0], observed[:, 1], s=6, c='lightgray', alpha=.35)
        mask = batch.pair_id == pair_id
        path_x = self._pair_path_coordinates(batch, pair_id)
        if path_x is not None:
            path_vis = self._map_positions(path_x)
            ax.plot(path_vis[:, 0], path_vis[:, 1], color='black', linewidth=1.2)
        start = batch.pair_x0_vis[pair_id]
        end = batch.pair_x1_vis[pair_id]
        ax.scatter(start[0], start[1], c='blue', s=45, label='source')
        ax.scatter(end[0], end[1], c='red', s=45, label='target')
        points = ax.scatter(
            batch.x_vis[mask, 0],
            batch.x_vis[mask, 1],
            c=batch.alpha[mask],
            cmap='viridis',
            s=28,
            label='samples',
        )
        if show_velocity and np.any(mask):
            velocity = batch.u_vis[mask].copy()
            if normalize_velocity:
                norm = np.linalg.norm(velocity, axis=1, keepdims=True)
                velocity = velocity / np.maximum(norm, 1e-8)
            ax.quiver(
                batch.x_vis[mask, 0],
                batch.x_vis[mask, 1],
                velocity[:, 0],
                velocity[:, 1],
                color='black',
                alpha=.55,
                scale=30 if normalize_velocity else None,
                width=.002,
            )
        ax.figure.colorbar(points, ax=ax, label='alpha')
        ax.legend()
        ax.set_title(f'V-centric OT pair {pair_id}')
        return ax.figure, ax

    def _pair_path_coordinates(self, batch, pair_id):
        transition = batch.metadata['pair_transition_idx'][pair_id]
        path = batch.metadata['paths'][pair_id]
        if len(path) < 2:
            return None
        x_pair = np.concatenate([
            self.sampler.X[transition],
            self.sampler.X[transition + 1],
        ])
        return x_pair[path]
