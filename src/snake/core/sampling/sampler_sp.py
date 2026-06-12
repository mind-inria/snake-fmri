from pathlib import Path
import copy
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf
from snake.core.sampling.samplers import NonCartesianAcquisitionSampler
from sparkling import Run
from sparkling.utils.naming import get_filename_from_params

def load_init_cfg_from_yaml(yaml_path: str | Path) -> dict:
    yaml_path = Path(yaml_path)
    try:
        import yaml  # pip install pyyaml
    except ImportError as e:
        raise ImportError("Missing PyYAML. Install with: pip install pyyaml") from e

    with yaml_path.open("r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("YAML root must be a dict.")
    return cfg


def inject_kinetic_constraints(cfg: dict) -> dict:
  
    from sparkling.constraints.utils import (
        first_derivative_transpose,
        first_derivative,
        proximity_L2,
        second_derivative,
    )

    cfg = copy.deepcopy(cfg)
    cfg["kinetic_constraint_init"] = {
        "speed": {
            "function": proximity_L2,
            "linear_op": first_derivative,
            "linear_adj_op": first_derivative_transpose,
        },
        "acceleration": {
            "function": proximity_L2,
            "linear_op": second_derivative,
            "linear_adj_op": second_derivative,
        },
    }
    return cfg

class SPARKLINGSampler(NonCartesianAcquisitionSampler):
    __sampler_name__ = "SPARKLING"
    __engine__ = "NUFFT"

    init_cfg_or_yaml: Any = None
    outdir: str = ""
    constant: bool = True
    verbose: bool = False
    num_shots: int | None = None
    rotate_angle: float = 0.0
    idx: int = 0
    clip_to_half: bool = True
    target_dwell_time_ms: float | None = None
    source_raster_time_ms: float | None = None

    def read_from_init_cfg(self):
        src = self.init_cfg_or_yaml
        if isinstance(src, (str, Path)) and str(src).endswith((".yml", ".yaml")):
            cfg = load_init_cfg_from_yaml(src)
            cfg = inject_kinetic_constraints(cfg)
            self.init_cfg = cfg
        elif isinstance(src, DictConfig):
            self.init_cfg = OmegaConf.to_container(src, resolve=True)
        elif isinstance(src, dict):
            self.init_cfg = copy.deepcopy(src)
        else:
            raise TypeError("init_cfg_or_yaml must be a dict/DictConfig or a .yaml/.yml path.")

    @staticmethod
    def _maybe_drop_terminal_sample(
        shots: np.ndarray,
        init_cfg: dict,
    ) -> np.ndarray:
        """Align SPARKLING shot length with writer/reader conventions.

        `runObj.current["shots"]` often contains one extra terminal sample
        compared to what `read_trajectory` returns from the saved `.bin`.
        """
        traj_params = init_cfg.get("traj_params", {})
        expected = traj_params.get("num_samples_per_shot")
        if expected is None:
            return shots

        expected = int(expected)
        if expected <= 1:
            return shots

        target = expected - 1
        if shots.shape[1] == target + 1:
            return shots[:, :-1, :]
        if shots.shape[1] > target:
            return shots[:, :target, :]
        return shots

    def _resample_to_dwell(
        self,
        shots: np.ndarray,
        sim_conf,
        init_cfg: dict,
    ) -> np.ndarray:
        source_raster = self.source_raster_time_ms
        if source_raster is None:
            source_raster = init_cfg.get("scan_consts", {}).get(
                "gradient_raster_time", sim_conf.hardware.raster_time_ms
            )
        target_dwell = self.target_dwell_time_ms
        if target_dwell is None:
            target_dwell = sim_conf.hardware.dwell_time_ms

        source_raster = float(source_raster)
        target_dwell = float(target_dwell)
        if target_dwell <= 0:
            raise ValueError("target_dwell_time_ms must be > 0.")

        new_len = int(round(shots.shape[1] * (source_raster / target_dwell)))
        new_len = max(new_len, 1)
        if new_len == shots.shape[1]:
            return shots

        sample_old = np.arange(shots.shape[1], dtype=np.float32)
        sample_new = np.linspace(0, shots.shape[1] - 1, new_len, dtype=np.float32)
        return np.stack(
            [
                np.stack(
                    [
                        np.interp(sample_new, sample_old, shots[i, :, d])
                        for d in range(shots.shape[2])
                    ],
                    axis=-1,
                )
                for i in range(shots.shape[0])
            ],
            axis=0,
        )
    def get_next_frame(self, sim_conf):
        """Generate the next frame."""
        if self.constant:
            if self._frame is None:
                self._frame = self._single_frame(sim_conf)
            return self._frame
        self.idx += 1
        return self._single_frame(sim_conf)
    def _single_frame(self, sim_conf):
        self.read_from_init_cfg()  # read configurations
        self.out_dir = Path(self.outdir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        INITIALIZATION = copy.deepcopy(self.init_cfg)
        if not self.constant:
            INITIALIZATION["traj_params"]["global_rot"] = self.rotate_angle * self.idx
        if self.num_shots is not None:
            INITIALIZATION["traj_params"]["num_shots"] = self.num_shots

        INIT = INITIALIZATION
        runObj = Run(**INIT, verbose=self.verbose)
        filename = get_filename_from_params(runObj, str(self.out_dir))
        runObj.initialize_shots()
        runObj.start_optimization(filename, INIT, self.verbose)

        shots = np.asarray(runObj.current["shots"] * np.pi, dtype=np.float32)
        shots = self._maybe_drop_terminal_sample(shots, INIT)
        new_shots = self._resample_to_dwell(shots, sim_conf, INIT)
        if self.clip_to_half:
            new_shots = np.clip(new_shots, -0.5, 0.5)
        return np.asarray(new_shots, dtype=np.float32)
