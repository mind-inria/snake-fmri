"""Sampling pattern generations."""

from __future__ import annotations
import logging
from typing import ClassVar
from typing_extensions import dataclass_transform
from numpy.typing import NDArray

from snake._meta import MetaDCRegister
from ..simulation import SimConfig

import ismrmrd as mrd


@dataclass_transform(kw_only_default=True)  # Required here for pyright to work.
class MetaSampler(MetaDCRegister):
    """MetaClass for Samplers."""

    dunder_name = "sampler"


class BaseSampler(metaclass=MetaSampler):
    """Sampler Interface.

    A Sampler is designed to generate a sampling pattern.
    """

    __sampler_name__: ClassVar[str]
    __engine__: ClassVar[str]
    __registry__: ClassVar[dict[str, type[BaseSampler]]]
    constant: bool = True

    def __post_init__(self):
        self._frame = None

    @property
    def log(self) -> logging.Logger:
        """Get a logger."""
        return logging.getLogger(f"simulation.samplers.{self.__class__.__name__}")

    def _single_frame(self, sim_conf: SimConfig) -> NDArray:
        # Generate a single frame
        raise NotImplementedError

    def get_next_frame(self, sim_conf: SimConfig) -> NDArray:
        """Generate the next frame."""
        if self.constant:
            if self._frame is None:
                self._frame = self._single_frame(sim_conf)
            return self._frame

        return self._single_frame(sim_conf)

    def add_all_acq_mrd(self, dataset: mrd.Dataset, sim_conf: SimConfig) -> mrd.Dataset:
        """Add all the acquisition to the dataset."""
        raise NotImplementedError

    def TR_vol_ms(self, sim_conf: SimConfig) -> float:
        """Get the TR in milliseconds."""
        raise NotImplementedError
