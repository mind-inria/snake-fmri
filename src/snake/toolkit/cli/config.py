"""Configuration of SNAKE using Hydra."""

from pathlib import Path
from typing import Any
from collections.abc import Callable
from dataclasses import dataclass, field
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, DictConfig

from snake.core.simulation import SimConfig
from snake.core.phantom.static import TissueFile
from snake.core.handlers import AbstractHandler
from snake.core.sampling import BaseSampler

from snake.toolkit.reconstructors import BaseReconstructor

# Import all handlers and samplers in plugins files
# to register them in their respective registries
# This is needed to be able to use them in the configuration file
# and to be able to use them in the CLI

import pkgutil
import importlib
import sys
import os

print("Importing plugins")
# Adding the current directory to the path
sys.path.insert(0, os.getcwd())
for finder, module_name, _ispkg in pkgutil.iter_modules():
    if module_name.startswith("snake_"):
        importlib.import_module(module_name)
        print(f"Imported {module_name} from {finder.path}")


@dataclass
class EngineConfig:
    """Engine configuration for SNAKE."""

    n_jobs: int = 1
    chunk_size: int = 1
    model: str = "simple"
    snr: float = 0
    nufft_backend: str = "finufft"
    slice_2d: bool = False


@dataclass
class PhantomConfig:
    """PhantomConfig."""

    name: str = "brainweb"
    sub_id: int = 4
    tissue_select: list[str] = field(default_factory=list)
    tissue_ignore: list[str] = field(default_factory=list)
    tissue_file: str | TissueFile = TissueFile.tissue_1T5


@dataclass
class StatConfig:
    """Statistical configuration for SNAKE."""

    roi_tissue_name: str = "ROI"
    roi_threshold: float = 0.5
    event_name: str = "block_on"


@dataclass
class ConfigSNAKE:
    """Configuration for SNAKE."""

    handlers: Any
    sampler: Any
    reconstructors: Any
    sim_conf: SimConfig = field(default_factory=SimConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    phantom: PhantomConfig = field(default_factory=PhantomConfig)
    stats: StatConfig = field(default_factory=StatConfig)
    cache_dir: Path = "${oc.env:PWD}/cache"  # type: ignore
    result_dir: Path = "${oc.env:PWD}/results"  # type: ignore
    filename: Path = "test.mrd"  # type: ignore


def conf_validator(cfg: DictConfig) -> ConfigSNAKE:
    """Validate the simulation configuration."""
    cfg_obj: ConfigSNAKE = OmegaConf.to_object(cfg)

    cfg_obj.sim_conf.fov.size = tuple(cfg_obj.sim_conf.fov.size)
    cfg_obj.sim_conf.fov.res_mm = tuple(cfg_obj.sim_conf.fov.res_mm)

    return cfg_obj


# Custom Resolver for OmegaConf
# allows to do:
#     _target_: {$snake.handler:motion-image}
# instead of hardcoding the path to the class


def snake_handler_resolver(name: str) -> str:
    """Get Custom resolver for OmegaConf to get handler name."""
    from snake.core.handlers import H

    cls = H[name]
    return cls.__module__ + "." + cls.__name__


def snake_sampler_resolver(name: str) -> str:
    """Get Custom resolver for OmegaConf to get handler name."""
    from snake.core.sampling import BaseSampler

    cls = BaseSampler.__registry__[name]
    return cls.__module__ + "." + cls.__name__


OmegaConf.register_new_resolver("snake.handler", snake_handler_resolver)
OmegaConf.register_new_resolver("snake.sampler", snake_sampler_resolver)


cs = ConfigStore.instance()
cs.store(name="base_config", node=ConfigSNAKE)

for handler_name, h_cls in AbstractHandler.__registry__.items():
    cs.store(group="handlers", name=handler_name, node={handler_name: h_cls})

for sampler, s_cls in BaseSampler.__registry__.items():
    cs.store(group="sampler", name=sampler, node={sampler: s_cls})

for reconstructor, r_cls in BaseReconstructor.__registry__.items():
    cs.store(group="reconstructors", name=reconstructor, node={reconstructor: r_cls})


def cleanup_cuda() -> None:
    """Cleanup CUDA."""
    import cupy as cp

    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    cp._default_memory_pool = cp.cuda.MemoryPool()
    cp._default_pinned_memory_pool = cp.cuda.PinnedMemoryPool()


def make_hydra_cli(fun: Callable) -> Callable:
    """Create a Hydra CLI for the function."""
    return hydra.main(
        version_base=None, config_path="../../../cli-conf", config_name="config"
    )(fun)
