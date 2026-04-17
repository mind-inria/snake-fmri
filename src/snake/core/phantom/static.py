"""Module to create phantom for simulation."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import logging
import os
from collections.abc import Generator
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from _typeshed import GenericPath

    from snake.mrd_utils.loader import MRDLoader

import ismrmrd as mrd
import numpy as np
from nibabel.nifti1 import Nifti1Image
from numpy.typing import NDArray

from snake._meta import ThreeFloats, ThreeInts

from ..off_resonance import get_field_map
from ..parallel import SharedArray, run_parallel
from ..simulation import SimConfig
from ..smaps import get_smaps
from ..transform import apply_affine4d, serialize_array, unserialize_array
from .contrast import _contrast_gre
from .utils import PropTissueEnum, TissueFile, parse_tissue_file, resize_tissues

log = logging.getLogger(__name__)

SNAKE_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "snake-fmri")

PHANTOM_ARRAYS = ["masks", "props", "labels", "smaps", "affine"]


@dataclass
class Phantom:
    """A Phantom consist of all spatial maps that are used in the simulation.

    It is a dataclass that contains the tissue masks, properties, labels, and
    spatial maps of the phantom.

    The tissue masks are a 3D array with the shape (n_tissues, x, y, z), where
    n_tissues is the number of tissues.

    The properties are a 2D array with the shape (n_tissues, n_properties), where
    n_properties is the number of properties.

    The labels are a 1D array with the shape (n_tissues,) containing the names
    of the tissues.

    The sensitivity maps are a 4D array with the shape (n_coils, x, y, z), where
    n_coils is the number of coils.

    The affine matrix is a 2D array with the shape (4, 4) containing the affine
    transformation matrix.

    """

    # TODO Add field map inhomogeneity in the phantom
    name: str
    masks: NDArray[np.float32]
    labels: NDArray[np.str_]
    props: NDArray[np.float32]
    smaps: NDArray[np.complex64] | None = None
    affine: NDArray[np.float32] = field(
        default_factory=lambda: np.eye(4, dtype=np.float32)
    )

    def add_tissue(
        self,
        tissue_name: str,
        mask: NDArray[np.float32],
        props: NDArray[np.float32],
        phantom_name: str | None = None,
    ) -> Phantom:
        """Add a tissue to the phantom. Creates a new Phantom object."""
        masks = np.concatenate((self.masks, mask[None, ...]), axis=0)
        labels = np.concatenate((self.labels, np.array([tissue_name])))
        props = np.concatenate((self.props, props), axis=0)
        return Phantom(
            phantom_name or self.name,
            masks,
            labels,
            props,
            smaps=self.smaps,
            affine=self.affine,
        )

    @property
    def labels_idx(self) -> dict[str, int]:
        """Get the index of the labels."""
        return {label: i for i, label in enumerate(self.labels)}

    def make_smaps(
        self,
        n_coils: int | None = None,
        sim_conf: SimConfig | None = None,
        antenna: str = "birdcage",
    ) -> None:
        """Get coil sensitivity maps for the phantom."""
        if n_coils is None:
            if sim_conf is None:
                raise ValueError("Either n_coils or sim_conf must be provided.")
            n_coils = sim_conf.hardware.n_coils
        if n_coils == 1:
            log.warning("Only one coil, no need for smaps.")
        elif n_coils > 1 and self.smaps is None:
            self.smaps = get_smaps(self.anat_shape, n_coils=n_coils, antenna=antenna)
            log.debug(f"Created smaps for {n_coils} coils.")
        elif self.smaps is not None:
            log.warning("Smaps already exists.")

    @classmethod
    def from_brainweb(
        cls,
        sub_id: int,
        sim_conf: SimConfig,
        tissue_file: str | TissueFile = TissueFile.tissue_1T5,
        tissue_select: list[str] | None = None,
        tissue_ignore: list[str] | None = None,
        output_res: float | ThreeFloats = 0.5,
        cache_dir: str | None = None,
    ) -> Phantom:
        """Get the Brainweb Phantom.

        Parameters
        ----------
        sub_id: int
            Subject ID of the brainweb dataset.
        sim_conf: SimConfig
            Simulation configuration.
        tissue_file: str
            File with the tissue properties.
        tissue_select: list[str]
            List of tissues to select.
        tissue_ignore: list[str]
            List of tissues to ignore.
        output_res: float
            Resolution of the output phantom.

        Returns
        -------
        Phantom
            The phantom object.
        """
        if cache_dir is None:
            cache_dir = os.environ.get("SNAKE_CACHE_DIR", SNAKE_CACHE_DIR)
        phantom_hash = hashlib.md5(
            json.dumps(
                dict(
                    sub_id=sub_id,
                    tissue_file=tissue_file,
                    tissue_select=tissue_select,
                    tissue_ignore=tissue_ignore,
                    output_res=output_res,
                    n_coils=sim_conf.hardware.n_coils,
                    fov=asdict(sim_conf.fov),
                ),
                sort_keys=True,
            ).encode()
        ).hexdigest()
        phantom_file = None
        if cache_dir is not False:
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            phantom_file = os.path.join(cache_dir, f"phantom_{phantom_hash}.npy")
            if os.path.exists(phantom_file):
                log.debug(f"Loading phantom from cache: {phantom_file}")
                return cls.from_mrd_dataset(phantom_file)

        from brainweb_dl import BrainWebTissuesV2, get_mri

        if tissue_ignore and tissue_select:
            raise ValueError("Only one of tissue_select or tissue_ignore can be used.")
        if tissue_select:
            tissue_select = [t.lower() for t in tissue_select]
        if tissue_ignore:
            tissue_ignore = [t.lower() for t in tissue_ignore]

        # TODO: Add A caching for the phantom. Use the SNAKE_CACHE_DIR env variable

        tissues_mask, affine = get_mri(sub_id, contrast="fuzzy", with_affine=True)
        tissues_mask = tissues_mask.astype(np.float32)
        affine = affine.astype(np.float32)
        tissues_mask = np.ascontiguousarray(tissues_mask.T)
        tissues_list = dict()
        select = []
        for name, props in parse_tissue_file(tissue_file).items():
            if (
                (tissue_select and name in tissue_select)
                or (tissue_ignore and name not in tissue_ignore)
                or (not tissue_select and not tissue_ignore)
            ):
                tissues_list[name] = props
                select.append(BrainWebTissuesV2[name.upper()])
        log.info(
            f"Selected tissues: {select}, {[t[0] for t in tissues_list]}",
        )
        if len(tissues_list) == 0:
            raise ValueError("No tissues selected")
        tissues_mask = tissues_mask[select]
        shape = tissues_mask.shape
        #
        # TODO: Use the sim shape properly.
        if output_res != 0.5:
            if isinstance(output_res, int | float):
                output_res = [output_res] * 3
            z = np.array([0.5, 0.5, 0.5]) / np.array(output_res)
            new_shape = (shape[0], *np.round(np.array(shape[1:]) * z).astype(int))
            tissue_resized = np.zeros(new_shape, dtype=np.float32)
            for i in range(3):
                affine[i, i] = output_res[i]
            run_parallel(
                resize_tissues,
                tissues_mask,
                tissue_resized,
                parallel_axis=0,
                z=tuple(z),
            )
            tissues_mask = tissue_resized

        smaps = None
        if sim_conf.hardware.n_coils > 1:
            smaps = get_smaps(
                tissues_mask.shape[1:],
                n_coils=sim_conf.hardware.n_coils,
            )

        phantom = cls(
            f"brainweb-{sub_id:02d}",
            tissues_mask,
            labels=np.array(tissues_list.keys()),
            props=np.array(tissues_list.values()),
            smaps=smaps,
            affine=affine,
        )

        if phantom_file:
            log.debug(f"Saving phantom to cache: {phantom_file}")
            phantom.to_mrd_dataset(phantom_file)
        return phantom

    @classmethod
    def from_mrd_dataset(
        cls, dataset: MRDLoader | os.PathLike | str, imnum: int = 0
    ) -> Phantom:
        """Load the phantom from a mrd dataset."""
        from snake.mrd_utils.loader import get_affine_from_image, MRDLoader

        if not isinstance(dataset, MRDLoader):
            dataset = MRDLoader(dataset)
        with dataset:
            image = dataset._read_image("phantom", imnum)
            name = image.meta.pop("name")
            labels = np.array(image.meta["labels"].split(","))
            props = unserialize_array(image.meta["props"])

            affine = get_affine_from_image(image)
            # smaps
            try:
                smaps = dataset._read_image("smaps", imnum).data
            except LookupError:
                smaps = None

        return cls(
            masks=image.data,
            labels=labels,
            props=props,
            name=name,
            affine=affine,
            smaps=smaps,
        )

    def to_mrd_dataset(self, dataset: mrd.Dataset | GenericPath) -> mrd.Dataset:
        """Add the phantom as an image to the dataset."""
        # Create the image
        if not isinstance(dataset, mrd.Dataset):
            dataset = mrd.Dataset(dataset, create_if_needed=True)

        meta_sr = mrd.Meta(
            {
                "name": self.name,
                "labels": f"{','.join(self.labels)}",
                "props": serialize_array(self.props),
                "affine": serialize_array(self.affine),
            }
        ).serialize()

        # Convert the affine matrix to position, field of view, etc.
        offsets = self.affine[:3, 3]
        res_mm = np.sqrt(np.sum(self.affine[:3, :3] ** 2, axis=0))
        position = (-offsets[0], -offsets[1], offsets[2])
        read_dir = self.affine[:3, 0] / res_mm[0]
        read_dir = (-read_dir[0], -read_dir[1], read_dir[2])
        phase_dir = self.affine[:3, 1] / res_mm[1]
        phase_dir = (-phase_dir[0], -phase_dir[1], phase_dir[2])
        slice_dir = self.affine[:3, 2] / res_mm[2]
        slice_dir = (-slice_dir[0], -slice_dir[1], slice_dir[2])

        fov_mm = tuple(np.array(self.anat_shape).astype(np.float32) * res_mm)
        base_header = mrd.image.ImageHeader(
            matrix_size=self.anat_shape,
            field_of_view=fov_mm,
            position=position,
            phase_dir=phase_dir,
            slice_dir=slice_dir,
            read_dir=read_dir,
            channels=1,
            acquisition_time_stamp=0,
            attribute_string_len=len(meta_sr),
        )
        header_phantom = copy.deepcopy(base_header)
        header_phantom.channels = self.n_tissues
        # Add the phantom data
        dataset.append_image(
            "phantom",
            mrd.image.Image(
                head=header_phantom,
                data=self.masks,
                attribute_string=meta_sr,
            ),
        )
        # Add the smaps
        if self.smaps is not None:
            header_smaps = copy.deepcopy(base_header)
            header_smaps.channels = len(self.smaps)
            dataset.append_image(
                "smaps",
                mrd.image.Image(
                    head=header_smaps,
                    data=self.smaps,
                ),
            )
        return dataset

    @classmethod
    @contextlib.contextmanager
    def from_shared_memory(
        cls,
        name: str,
        shared_arrays: dict[str, SharedArray | None],
    ) -> Generator[Phantom, None, None]:
        """Give access the tissue masks and properties in shared memory."""
        arrays = {}
        with contextlib.ExitStack() as stack:
            for arr_name in PHANTOM_ARRAYS:
                if (arr := shared_arrays.get(arr_name, None)) is not None:
                    arrays[arr_name] = stack.enter_context(arr.as_array())
            yield cls(
                name=name,
                **arrays,
            )

    def in_shared_memory(
        self, manager: SharedMemoryManager
    ) -> tuple[str, dict[str, SharedArray]]:
        """Create a dict that hold all the arrays of the phantom in shared memory."""
        shared_arrays = {}
        for arr_name in PHANTOM_ARRAYS:
            if arr := getattr(self, arr_name) is not None:
                shared_arrays[arr_name] = SharedArray.from_array(manager, arr)  # type: ignore
        return self.name, shared_arrays

    def masks2nifti(self) -> Nifti1Image:
        """Return the masks of the phantom as a Nifti object."""
        return Nifti1Image(
            self.masks,
            affine=self.affine,
            extra={"props": self.props, "labels": self.labels},
        )

    def smaps2nifti(self) -> Nifti1Image:
        """Return the smaps as a Nifti object."""
        if self.smaps is not None:
            return Nifti1Image(self.smaps, affine=self.affine)
        else:
            raise ValueError("No Smaps to convert.")

    def to_nifti(
        self, filename: str | GenericPath = None
    ) -> tuple[GenericPath, GenericPath | None]:
        """Save the phantom as a pair of niftis file."""
        mask_nifti = self.masks2nifti()
        smaps_nifti = None
        smaps_filename = None

        if self.smaps is not None:
            smaps_nifti = self.smaps2nifti()
            smaps_filename = Path(str(filename).replace(".nii", "_smaps.nii"))
            smaps_nifti.to_filename(smaps_filename)
        if not filename:
            return filename, smaps_nifti
        mask_nifti.to_filename(filename)
        return filename, smaps_filename

    @classmethod
    def from_nifti(
        cls,
        mask_nifti: Nifti1Image | GenericPath,
        props: NDArray[np.float32] | None = None,
        labels: NDArray[np.str_] | None = None,
        smaps: Nifti1Image | GenericPath | None = None,
    ) -> Phantom:
        """Create a phantom from nifti files."""
        if not isinstance(mask_nifti, Nifti1Image):
            mask_nifti_name = mask_nifti
            mask_nifti = Nifti1Image.from_filename(mask_nifti)
        else:
            mask_nifti_name = mask_nifti.get_filename() or "from_nifti"
        if smaps and not isinstance(smaps, Nifti1Image):
            smaps_nifti = Nifti1Image.from_filename(smaps)
        else:
            smaps_nifti = smaps
        affine = mask_nifti.affine
        if props is None:
            props = mask_nifti.extra["props"]
        if labels is None:
            labels = mask_nifti.extra["labels"]
        masks = np.asarray(mask_nifti.get_fdata()).astype(np.float32)
        smaps = None
        if isinstance(smaps_nifti, Nifti1Image):
            smaps = np.asarray(smaps_nifti.get_fdata()).astype(np.complex64)
        return cls(
            name=str(mask_nifti_name),
            masks=masks,
            labels=labels,
            props=props,
            smaps=smaps,
            affine=affine,
        )

    def contrast(
        self,
        *,
        TR: float | None = None,
        TE: float | None = None,
        FA: float | None = None,
        sequence: Literal["GRE"] = "GRE",
        sim_conf: SimConfig | None = None,
        resample: bool = True,
        aggregate: bool = True,
        use_gpu: bool = True,
    ) -> NDArray[np.float32]:
        """Compute the contrast of the phantom for a given sequence.

        Parameters
        ----------
        TR: float
        TE: float
        FA: float
        sim_conf: SimConfig
            Other way to provide sequence parameters
        aggregate: bool, optional default=True
            Sum all the tissues contrast for getting a single image.
        sequence="GRE"
            Default value, no other value is currently supported.

        Results
        -------
        NDArray
            The contrast of the tissues.
        """
        if resample:
            if sim_conf is None:
                raise ValueError("sim_conf must be provided for resampling.")
            affine = sim_conf.fov.affine
            shape = sim_conf.fov.shape
            self = self.resample(affine, shape, use_gpu=use_gpu)

        if TR is None or TE is None or FA is None:
            if sim_conf is not None:
                TR = sim_conf.seq.TR_eff  # Here we use the effective TR.
                TE = sim_conf.seq.TE
                FA = sim_conf.seq.FA
            else:
                raise ValueError("Missing either sim_conf or TR,TE,FA")

        if sequence.upper() == "GRE":
            contrasts = _contrast_gre(self.props, TR=TR, TE=TE, FA=FA)
        else:
            raise NotImplementedError("Contrast not implemented.")
        if aggregate:
            ret = np.zeros(self.anat_shape, dtype=np.float32)
            for c, m in zip(contrasts, self.masks, strict=False):
                ret += c * m
            return ret
        else:
            return self.masks * contrasts[(..., *([None] * len(self.anat_shape)))]

    def resample(
        self,
        new_affine: NDArray,
        new_shape: ThreeInts,
        use_gpu: bool = False,
        **kwargs: Any,
    ) -> Phantom:
        """Resample the phantom to a new shape and affine matrix.

        Parameters
        ----------
        new_affine : NDArray
            The new affine matrix.
        new_shape : ThreeInts
            The new shape of the phantom.
        use_gpu : bool, optional
            Use the GPU for the resampling, by default False.
        """
        new_masks = apply_affine4d(
            self.masks,
            old_affine=self.affine,
            new_affine=new_affine,
            new_shape=new_shape,
            use_gpu=use_gpu,
            **kwargs,
        )
        new_smaps = None
        if self.smaps is not None:
            new_smaps = apply_affine4d(
                self.smaps,
                old_affine=self.affine,
                new_affine=new_affine,
                new_shape=new_shape,
                use_gpu=use_gpu,
                **kwargs,
            )
        return Phantom(
            self.name,
            new_masks,
            self.labels,
            self.props,
            smaps=new_smaps,
            affine=new_affine,
        )

    @property
    def anat_shape(self) -> tuple[int, ...]:
        """Get the shape of the base volume."""
        return self.masks.shape[1:]

    @property
    def n_tissues(self) -> int:
        """Get the number of tissues."""
        return len(self.masks)

    def __repr__(self):
        ret = f"Phantom[{self.name}]: {self.props.shape}\n"
        ret += f"{'tissue name':14s}" + "".join(
            f"{str(prop):4s}" for prop in PropTissueEnum
        )
        ret += "\n"
        for i, tissue_name in enumerate(self.labels):
            props = self.props[i]
            ret += (
                f"{tissue_name:14s}"
                + "".join(f"{props[p]:4}" for p in PropTissueEnum.__members__.values())
                + "\n"
            )
        return ret

    def __deepcopy__(self, memo: Any) -> Phantom:
        """Create a copy of the phantom."""
        return Phantom(
            name=self.name,
            masks=deepcopy(self.masks, memo),
            labels=deepcopy(self.labels, memo),
            props=deepcopy(self.props, memo),
            smaps=deepcopy(self.smaps, memo),
            affine=deepcopy(self.affine, memo),
        )

    def copy(self) -> Phantom:
        """Return deep copy of the Phantom."""
        return deepcopy(self)
