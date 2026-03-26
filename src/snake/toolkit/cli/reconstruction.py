"""CLI for SNAKE."""

import gc
import json
import logging
import os
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf

from snake.mrd_utils import (
    CartesianFrameDataLoader,
    MRDLoader,
    NonCartesianFrameDataLoader,
    parse_sim_conf,
    read_mrd_header,
)
from snake.toolkit.analysis.stats import contrast_zscore, get_scores
from snake.toolkit.cli.config import cleanup_cuda, conf_validator, make_hydra_cli

log = logging.getLogger(__name__)


def reconstruction(cfg: DictConfig) -> None:
    """Reconstruction function."""
    # Disable HDF5 file locking
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    cfg = conf_validator(cfg)
    log.info("Starting Reconstruction")
    hdr = read_mrd_header(cfg.filename)
    *version, engine = hdr.acquisitionSystemInformation.systemModel.split("-")
    log.info(f"Data from {version}, using engine {engine}")

    # Extract sim_confg
    sim_conf = parse_sim_conf(hdr)
    if sim_conf != cfg.sim_conf:
        log.warning(
            "Loaded simulation configuration is different from the one in the config."
            " using the one from CLI."
        )
        log.warning("Loaded: %s", sim_conf)
        log.warning("Config: %s", cfg.sim_conf)
        sim_conf = cfg.sim_conf

    DataLoader: type[MRDLoader]
    if engine in ["EPI", "EVI"]:
        DataLoader = CartesianFrameDataLoader
    elif engine == "NUFFT":
        DataLoader = NonCartesianFrameDataLoader

    # Reconstructor.setup(sim_conf) # initialize operators
    # array = Reconstructor.reconstruct(dataloader, sim_conf)
    for name, rec in cfg.reconstructors.items():
        rec_str = str(rec)  # FIXME Also use parameters  of reconstructors
        data_rec_file = Path(f"data_rec_{rec_str}.npy")
        log.info(f"Using {name} reconstructor")
        with DataLoader(cfg.filename) as data_loader:
            # rec.setup(data_loader.get_sim_conf())
            rec_data = rec.reconstruct(data_loader)
        log.info(f"Reconstruction done with {name}")
        # Save the reconstruction
        if np.any(np.isnan(rec_data)):
            log.warning("NaNs detected in reconstruction")
        np.save(data_rec_file, rec_data)
        log.info(f"Saved to {data_rec_file.resolve()}")
    gc.collect()

    # Extract statistical data:
    with DataLoader(cfg.filename) as data_loader:
        phantom = data_loader.get_phantom()
        roi_mask = phantom.masks[phantom.labels == cfg.stats.roi_tissue_name]
        dyn_datas = data_loader.get_all_dynamic()
        waveform_name = f"activation-{cfg.stats.event_name}"
        good_d = None
        for d in dyn_datas:
            if d.name == waveform_name:
                good_d = d
                break
        if good_d is None:
            raise ValueError("No dynamic data found matching waveform name")

        bold_signal = good_d.data[0]
        bold_sample_time = np.arange(len(bold_signal)) * sim_conf.seq.TR / 1000
        del phantom
        del dyn_datas
    gc.collect()

    results = []
    for _name, rec in cfg.reconstructors.items():
        rec_str = str(rec)  # FIXME Also use parameters  of reconstructors
        data_rec_file = Path(f"data_rec_{rec_str}.npy").resolve()
        data_zscore_file = Path(f"data_zscore_{rec_str}.npy").resolve()
        rec_data = abs(np.load(data_rec_file))
        TR_vol = sim_conf.max_sim_time / len(rec_data)
        try:
            z_score = contrast_zscore(
                rec_data,
                TR_vol,
                bold_signal,
                bold_sample_time,
                cfg.stats.event_name,
            )
        except Exception as e:
            log.error(f"Error in contrast_zscore: {e}")
            continue
        stats_results = get_scores(z_score, roi_mask, cfg.stats.roi_threshold)
        np.save(data_zscore_file, z_score)

        # Reload the config from hydra and add it to the result file
        # This way OmegaConf does all the serialization for us.
        conf_dict = OmegaConf.load(Path.cwd() / ".hydra" / "config.yaml")
        conf_dict = OmegaConf.to_container(conf_dict)
        results.append(
            conf_dict
            | {
                "results": stats_results,
                "data_zscore": data_zscore_file,
                "data_rec": data_rec_file,
                "data_raw": cfg.filename.resolve(),
            }
        )

    with open("results.json", "w") as f:
        json.dump(results, f, default=lambda o: str(o))
    cleanup_cuda()


reconstruction_cli = make_hydra_cli(reconstruction)

if __name__ == "__main__":
    reconstruction_cli()
