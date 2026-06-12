"""Activation Handler."""

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from ...._meta import LogMixin
from ...phantom import Phantom, DynamicData
from ...simulation import SimConfig
from ..base import AbstractHandler
from ..utils import apply_weights
from .roi import BRAINWEB_OCCIPITAL_ROI, get_indices_inside_ellipsoid
from .bold import get_bold, block_design, get_event_ts
from ...transform import apply_affine


class ActivationMixin(LogMixin):
    """Add activation inside the region of interest. for a single type of event.

    Parameters
    ----------
    event_condition:
        array-like of shape (3, n_events)
        yields description of events for this condition as a
        (onsets, durations, amplitudes) triplet
    hrf_model:
        Choice for the HRF, FIR is not
    oversampling:
        Oversampling factor to perform the convolution. Default=50.
    min_onset:
        Minimal onset relative to frame_times[0] (in seconds)
        events that start before frame_times[0] + min_onset are not considered.
        Default=-24.
    base_tissue_name:
        Name of the tissue to intersect with the ROI.
    atlas: str, default=None
        Name of the atlas to use for the ROI.
    atlas_label: int | str, default=-1
        Label of the ROI in the atlas.


    Notes
    -----
    If no atlases is provided, the ROI is computed by intersecting the base tissue with
    an ellipsoid in the occipital region.
    If a probabilistic atlas is provided, the effective BOLD signal will be the product
    of the voxel base_tissue_name (e.g. gray-matter) and of the atlas mask.

    See Also
    --------
    nilearn.compute_regressors
    """

    event_condition: pd.DataFrame | np.ndarray
    duration: float
    offset: float = 0
    event_name: str
    roi_tissue_name: str = "ROI"
    delta_r2s: float = 1000.0  # mHz
    hrf_model: str = "glover"
    oversampling: int = 10
    min_onset: float = -24.0
    base_tissue_name: str = "gm"  # The ROI intersected with the gray matter mask.
    # Use nilearn for downloading the atlas.
    atlas: str | None = "hardvard-oxford__cort-maxprob-thr50-1mm"
    atlas_label: int | str = ""
    roi_threshold: float | None = None

    def get_static(self, phantom: Phantom, sim_config: SimConfig) -> Phantom:
        """Get the static ROI."""
        # shape: tuple[int, ...] | None = phantom.tissues_mask.shape[1:]
        tissue_index = phantom.labels == self.base_tissue_name
        if self.atlas is None:
            roi = self._get_roi_base(phantom)
        else:
            roi = self._get_roi_atlas(phantom)
        if self.roi_threshold is not None:
            roi = np.float32(roi > self.roi_threshold)

        # update the phantom
        new_phantom = phantom.add_tissue(
            self.roi_tissue_name,
            roi,
            phantom.props[tissue_index, :],
            phantom.name + "-roi",
        )

        return new_phantom

    def _get_roi_base(self, phantom: Phantom) -> NDArray:
        tissue_index = phantom.labels == self.base_tissue_name
        if tissue_index.sum() == 0:
            raise ValueError(
                f"Tissue {self.base_tissue_name} not found in the phantom."
            )
        roi_base = phantom.masks[tissue_index].squeeze().copy()
        occ_roi = BRAINWEB_OCCIPITAL_ROI.copy()
        roi_zoom = np.array(roi_base.shape) / np.array(occ_roi["shape"])
        self.log.debug(
            "ROI parameters (orig, target, zoom) %s, %s, %s",
            occ_roi["shape"],
            roi_base.shape,
            roi_zoom,
        )
        ellipsoid = get_indices_inside_ellipsoid(
            roi_base.shape,
            center=np.array(occ_roi["center"]) * roi_zoom,
            semi_axes_lengths=np.array(occ_roi["semi_axes_lengths"]) * roi_zoom,
            euler_angles=occ_roi["euler_angles"],
        )
        roi_base[~ellipsoid] = 0

        return roi_base

    def _get_roi_atlas(self, phantom: Phantom) -> NDArray:
        """Get the ROI from the atlas.

        Currently, only the Harvard-Oxford atlas is supported.
        """
        atlas_base, atlas_name = self.atlas.split("__")
        from nilearn.datasets.atlas import fetch_atlas_harvard_oxford

        if atlas_base == "hardvard-oxford":
            atlas = fetch_atlas_harvard_oxford(atlas_name=atlas_name) 
        else:
            raise ValueError(f"Atlas {atlas_base} not supported.")
        maps = atlas.maps
        if isinstance(self.atlas_label, str):
            idx = atlas.labels.index(self.atlas_label)
        else:
            idx = self.atlas_label
        if maps.dataobj.ndim == 4:  # probabilistic atlas
            atlas_mask = np.array(maps.dataobj[..., idx]).astype(np.float32)
        else:
            atlas_mask = np.array(maps.dataobj == idx).astype(np.float32)
        # Resample the atlas to the phantom affine
        atlas_mask = apply_affine(
            atlas_mask,
            old_affine=maps.affine,
            new_affine=phantom.affine,
            new_shape=phantom.anat_shape,
            use_gpu=True,
        )
        roi = atlas_mask * phantom.masks[phantom.labels_idx[self.base_tissue_name]]
        return roi

    def get_dynamic(self, phantom: Phantom, sim_conf: SimConfig) -> DynamicData:
        """Get dynamic time series for adding Activations."""
        bold_strength = sim_conf.seq.TE / self.delta_r2s

        self.log.info("Computed BOLD Strength: %s", bold_strength)
        bold = get_bold(
            sim_conf.sim_tr_ms,
            sim_conf.max_sim_time,
            self.event_condition,
            self.hrf_model,
            self.oversampling,
            self.min_onset,
            bold_strength,
        ).squeeze()
        events = get_event_ts(
            self.event_condition,
            sim_conf.max_sim_time,
            sim_conf.sim_tr_ms,
            self.min_onset,
        ).squeeze()
        return DynamicData(
            name="-".join(["activation", self.event_name]),
            data=np.concatenate([bold[None, :], events[None, :]]),
            func=self.apply_weights,
        )

    @staticmethod
    def apply_weights(phantom: Phantom, data: NDArray, time_idx: int) -> Phantom:
        """Apply weights to the ROI."""
        return apply_weights(phantom, "ROI", data[0], time_idx)


class BlockActivationHandler(ActivationMixin, AbstractHandler):
    """Activation Handler with block design.

    Parameters
    ----------
    block_on: float
        time the block activation is on.
    block_off: float
        time the block activation is off
    duration: float
        Total duration of the pattern in seconds
    offset: float, default 0
        Start time of the pattern in seconds
    roi_tissue_name: str, default "ROI"
        Name of the ROI tissue
    event_name: str, default "block_on"
        Name of the event
    delta_r2s: float, default 1000.0
        Delta R2s value
    hrf_model: str, default "glover"
        HRF model
    oversampling: int, default 50
        Oversampling factor
    min_onset: float, default -24.0
        Minimal onset
    roi_threshold: float, default 0.0
        ROI threshold

    Notes
    -----
    See Also the GLM module of Nilearn.


    """

    __handler_name__ = "activation-block"

    block_on: float
    block_off: float
    duration: float
    offset: float = 0
    roi_tissue_name: str = "ROI"
    event_name: str = "block_on"
    delta_r2s: float = 1000.0
    hrf_model: str = "glover"
    oversampling: int = 50
    min_onset: float = -24.0
    base_tissue_name: str = "gm"
    atlas: str | None = "hardvard-oxford__cort-maxprob-thr50-1mm"
    atlas_label: int | str = 48
    roi_threshold: float | None = None

    def __post_init__(self):
        self.event_condition = block_design(
            self.block_on,
            self.block_off,
            self.duration,
            self.offset,
            self.event_name,
        )
