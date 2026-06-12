"""Plotting utilities for the project."""

import matplotlib
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from mpl_toolkits.axes_grid1.axes_divider import Size, make_axes_locatable
from skimage.measure import find_contours
from matplotlib.cm import ScalarMappable, _colormaps


def get_coolgraywarm(thresh: float = 3, max: float = 7) -> matplotlib.colorbar.Colorbar:
    """Get a cool-warm colorbar, with gray inside the threshold."""
    coolwarm = matplotlib.colormaps["coolwarm"].resampled(256)
    newcolors = coolwarm(np.linspace(0, 1, 256))
    gray = np.array([0.8, 0.8, 0.8, 1])
    minthresh = int(128 + (thresh / max) * 128)
    maxthresh = int(128 - (thresh / max) * 128)
    newcolors[minthresh:maxthresh, :] = gray
    cool_gray_warm = matplotlib.colors.ListedColormap(newcolors)
    return cool_gray_warm


# %%
def _get_axis_properties(
    array_bg: NDArray,
    cuts: tuple[int, ...],
    width_inches: float,
    cbar: bool = True,
    arr_pad: int = 4,
    tight_crop: bool = True,
) -> tuple[
    NDArray,
    NDArray,
    tuple[tuple[slice, slice], ...],
    tuple[tuple[Any, Any, Any], ...],
]:
    """Generate mplt toolkit axes dividers for a 3D array.

    Parameters
    ----------
    array_bg: 3D array
        The 3D array to display.
    cuts: tuple
        The cuts to performs to create 3 2D array to display.
    width_inches: float
        The width of the figure in inches.
    cbar: bool
        Display the colorbar.
    arr_pad: int
        Padding to add to the bounding box.
    tight_crop: bool, default True
        If True, crop the image to their bounding box, to remove empty space.

    Returns
    -------
    hdiv: np.ndarray
        The horizontal division.
    vdiv: np.ndarray
        The vertical division.
    bbox: tuple
        The bounding box of the 2D arrays cuts.
    slices: tuple
        The slices to take from the 3D array.
    """
    slices = (np.s_[cuts[0], :, :], np.s_[:, cuts[1], :], np.s_[:, :, cuts[2]])
    bbox: list[tuple] = [(None, None) for _ in range(3)]
    for i in range(3):
        cut = array_bg[slices[i]]
        if cut.dtype != "bool":
            mask = abs(cut) > 0.5 * np.percentile(abs(cut), 95)
        else:
            mask = cut
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        try:
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
        except IndexError:
            rmin, rmax = 0, mask.shape[0]
            cmin, cmax = 0, mask.shape[1]
        if tight_crop:
            rmin = max(0, rmin - arr_pad)
            rmax = min(rmax + arr_pad, mask.shape[0])
            cmin = max(0, cmin - arr_pad)
            cmax = min(cmax + arr_pad, mask.shape[1])
            bbox[i] = (slice(rmin, rmax), slice(cmin, cmax))
        else:
            bbox[i] = (slice(0, cut.shape[0]), slice(0, cut.shape[1]))
    hdiv, vdiv = _get_hdiv_vdiv(array_bg, bbox, slices, width_inches, cbar=cbar)

    return hdiv, vdiv, tuple(bbox), slices


def _get_hdiv_vdiv(
    array_bg: NDArray,
    bbox: tuple[tuple[slice]],
    slices: tuple[slice],
    width_inches: float,
    cbar: bool = False,
) -> tuple[NDArray, NDArray]:
    sizes = np.array([(bb.stop - bb.start) for b in bbox for bb in b])

    sizes = tuple(array_bg[s][b].shape for s, b in zip(slices, bbox, strict=False))
    alpha1 = sizes[1][1] / sizes[2][1]
    update_sizes = [[0, 0], [0, 0], [0, 0]]
    update_sizes[2][0] = sizes[2][0]
    update_sizes[2][1] = sizes[2][1]
    alpha1 = sizes[2][1] / sizes[1][1]
    update_sizes[1][0] = sizes[1][0] * alpha1
    update_sizes[1][1] = sizes[1][1] * alpha1
    alpha2 = (update_sizes[2][0] + update_sizes[1][0]) / sizes[0][0]
    update_sizes[0][0] = sizes[0][0] * alpha2
    update_sizes[0][1] = sizes[0][1] * alpha2

    aspect = update_sizes[0][0] / (update_sizes[0][1] + update_sizes[1][1])
    split_lr = update_sizes[0][1] / (update_sizes[1][1] + update_sizes[0][1])
    split_tb = update_sizes[1][0] / (update_sizes[1][0] + update_sizes[2][0])
    hdiv = [
        width_inches * split_lr,
        width_inches * (1 - split_lr),
    ]

    if cbar:
        hdiv.extend(
            [
                0.02 * hdiv[0],
                0.02 * hdiv[0],
            ]
        )
    np.array(hdiv)
    height_inches = width_inches * aspect
    vdiv = np.array([height_inches * split_tb, height_inches * (1 - split_tb)])
    return hdiv, vdiv


def get_mask_cuts_mask(mask: NDArray) -> tuple[int, ...]:
    """Get the optimal cut that expose maximum number of voxel in mask."""
    max_cuts = [0] * len(mask.shape)
    for i in range(len(max_cuts)):
        max_cuts[i] = int(np.argmax(np.sum(mask, axis=tuple(np.array([-2, -1]) + i))))
    return tuple(max_cuts)


def plot_frames_activ(
    background: NDArray,
    z_score: NDArray,
    rois: list[NDArray] | None,
    ax: plt.Axes,
    slices: tuple[Any, ...],
    bbox: tuple[Any, ...],
    z_thresh: float = 3,
    z_max: float = 11,
    bg_cmap: str = "gray",
    roi_colors: list[str] | None = None,
) -> tuple[plt.Axes, matplotlib.image.AxesImage]:
    """Plot activation maps and background.

    Parameters
    ----------
    background: 3D array
    z_score: 3D array
    roi: 3D array
    ax: plt.Axes
    slices: tuple
    bbox: tuple
    z_thresh: float
    z_max: float
    bg_cmap: str
    """
    bg = background[slices][bbox].squeeze()
    im = ax.imshow(
        bg,
        vmin=np.min(background),
        vmax=np.max(background),
        cmap=bg_cmap,
        origin="lower",
        aspect="equal",
    )
    if z_score is not None:
        masked_z = z_score[slices][bbox].squeeze()
        masked_z[masked_z < z_thresh] = np.nan
        im = ax.imshow(
            masked_z,
            alpha=1,
            cmap=get_coolgraywarm(z_thresh, max=z_max),
            vmin=-z_max,
            vmax=z_max,
            aspect="equal",
            interpolation="nearest",
            origin="lower",
        )
    if rois is not None:
        if roi_colors is None:
            roi_colors = ["r", "g", "b", "y"][: len(rois)]
        for roi, color in zip(rois, roi_colors):
            roi_cut = roi[slices][bbox].squeeze()
            contours = find_contours(roi_cut)
            for c in contours:
                ax.plot(c[:, 1], c[:, 0], c=color, label="roi", linewidth=1)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax, im


def axis3dcut(
    background: NDArray[np.float32],
    z_score: NDArray[np.float32] | None,
    gt_roi: NDArray | None = None,
    width_inches: float = 7,
    cbar: bool = True,
    cuts: tuple[int, ...] | tuple[float, ...] | None = None,
    bbox: tuple[tuple[Any, Any], ...] | None = None,
    slices: tuple[tuple[Any, Any, Any], ...] | None = None,
    bg_cmap: str = "gray",
    ax: plt.Axes | None = None,
    vmin_vmax: tuple[float] = None,
    z_thresh: float = 3,
    z_max: float = 11,
    tight_crop: bool = False,
    roi_colors: list[str] | None = None,
) -> tuple[plt.Figure, plt.Axes, tuple[int, ...]]:
    """Display a 3D image with zscore and ground truth ROI.

    This function is used to display a 3D brain image with optional overlay for
    the z-score and the ground truth ROI outline.

    Parameters
    ----------
    background: 3D array
        The background image to display.
    z_score: 3D array, optional
        The z-score activation map to display, thresholded at z_thresh.
    gt_roi: 3D array, optional
        The ground truth ROI to display. If None, no ROI is displayed.
    width_inches: float, optional
        The width of the figure in inches.
    cbar: bool, optional
        Display the colorbar.
    cuts: tuple, optional
        The cuts to performs to create 3 2D array to display.
        If None, the cuts are computed, such that the ROI is maximally exposed.
    bbox: tuple, optional
        The bounding box to display.
    slices: tuple, optional
        The slices to display.
    bg_cmap: str, optional
        The colormap for the background image.
    ax: plt.Axes, optional
        The axes to use to display the image.
    vmin_vmax: tuple, optional
        The vmin and vmax to use for the background image.
    z_thresh: float, optional
        The threshold to use for the z-score.
    z_max: float, optional
        The maximum value for the z-score.
    tight_crop: bool, optional
        If True, crop the image to their bounding box, to remove empty space.

    Returns
    -------
    fig: plt.Figure
        The figure.
    ax: plt.Axes
        The axes.

    """
    #    ax.axis("off")
    if isinstance(gt_roi, np.ndarray):
        gt_roi = [gt_roi]
    if cuts is None and gt_roi is not None:
        cuts_ = get_mask_cuts_mask(gt_roi[0])
        gt_roi_ = gt_roi
    elif cuts is not None and gt_roi is not None:
        cuts_ = cuts
        gt_roi_ = gt_roi
    elif cuts is None and gt_roi is None:
        raise ValueError("Missing gt_roi to compute ideal cuts.")
    elif cuts is not None and gt_roi is None:
        cuts_ = cuts
        gt_roi_ = None

    if all(isinstance(c, float) and 0 < c < 1 for c in cuts_):
        cuts_ = tuple(round(c * background.shape[i]) for i, c in enumerate(cuts_))

    if bbox is None and slices is None:
        hdiv, vdiv, bbox_, slices_ = _get_axis_properties(
            background,
            cuts_,
            width_inches,
            cbar=cbar,
            tight_crop=tight_crop,
        )
    elif bbox is not None and slices is not None:
        hdiv, vdiv = _get_hdiv_vdiv(background, bbox, slices, width_inches, cbar=cbar)
        bbox_ = bbox
        slices_ = slices
    else:
        raise ValueError("Missing either bbox or slices.")

    if ax is not None:
        fig = ax.get_figure()
    else:
        # TODO Use the correct figure size
        fig, ax = plt.subplots(figsize=(width_inches, width_inches))

    divider = make_axes_locatable(ax)
    divider.set_horizontal([Size.Fixed(s) for s in hdiv])
    divider.set_vertical([Size.Fixed(s) for s in vdiv])
    axG: list[plt.Axes] = [None, None, None]
    for i, (nx, ny, ny1) in enumerate([(0, 0, 2), (1, 0, 1), (1, 1, 2)]):
        axG[i] = plt.Axes(fig, ax.get_position(original=True))
        axG[i].set_axes_locator(divider.new_locator(nx=nx, ny=ny, ny1=ny1))
        fig.add_axes(axG[i])
    for i in range(3):
        plot_frames_activ(
            background,
            z_score,
            gt_roi_,
            axG[i],
            slices_[i],
            bbox_[i],
            bg_cmap=bg_cmap,
            z_thresh=z_thresh,
            z_max=z_max,
            roi_colors=roi_colors,
        )

    if cbar:
        cax = type(ax)(fig, ax.get_position(original=True))
        cax.set_axes_locator(divider.new_locator(nx=3, ny=0, ny1=-1))
        if z_score is not None:
            im = ScalarMappable(norm="linear", cmap=get_coolgraywarm())
            im.set_clim(-z_max, z_max)
            matplotlib.colorbar.Colorbar(cax, im, orientation="vertical")
            cax.set_ylabel("z-scores", labelpad=-20)
            cax.set_yticks(
                np.concatenate(
                    [
                        -np.arange(3, z_max + 1, 2),
                        np.arange(3, z_max + 1, 2),
                    ]
                )
            )
        else:
            # use the background image
            if vmin_vmax is None:
                vmin, vmax = (np.min(background), np.max(background))
            else:
                vmin, vmax = vmin_vmax
            im = ScalarMappable(norm="linear", cmap=bg_cmap)
            im.set_clim(vmin=vmin, vmax=vmax)
            matplotlib.colorbar.Colorbar(cax, im, orientation="vertical")
        fig.add_axes(cax)

    ax.set_axes_locator(divider.new_locator(nx=0, ny=0, ny1=-1, nx1=-1))
    ax.set_zorder(10)
    ax.axis("off")
    # ax.set_xticks([])
    # ax.set_yticks([])
    return fig, ax, cuts_
