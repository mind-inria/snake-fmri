#!/usr/bin/env python
import numpy as np
import logging
from typing import Any
from mrinufft.io.utils import add_phase_to_kspace_with_shifts
from mrinufft.io.siemens import siemens_quat_to_rot_mat, nifti_affine
from mrinufft.io.nsp import read_arbgrad_rawdat, read_trajectory
from snake.mrd_utils.loader import MRDLoader
from snake.core.simulation import SimConfig, FOVConfig, HardwareConfig, GreConfig

log = logging.getLogger(__name__)


def parse_hdr_data(filename, removeOS=False, doAverage=True):
    """Parse Siemens raw data header using mapVBVD.

    Returns a dictionary with relevant header information
    and the mapVBVD object.
    """
    try:
        from mapvbvd import mapVBVD
    except ImportError as err:
        raise ImportError(
            "The mapVBVD module is not available. Please install "
            "it along with the [extra] dependencies "
            "or using `pip install pymapVBVD`."
        ) from err
    twixObj = mapVBVD(filename)
    if isinstance(twixObj, list):
        twixObj = twixObj[-1]
    twixObj.image.flagRemoveOS = removeOS
    twixObj.image.flagDoAverage = doAverage
    hdr = {
        "n_coils": int(twixObj.image.NCha),
        "n_shots": int(twixObj.image.NLin) * int(twixObj.image.NPar),
        "n_contrasts": int(twixObj.image.NSet),
        "n_adc_samples": int(twixObj.image.NCol),
        "n_slices": int(twixObj.image.NSli),
        "n_average": int(twixObj.image.NAve),
        "n_reps": int(twixObj.image.NRep),
        "orientation": siemens_quat_to_rot_mat(twixObj.image.slicePos[0][-4:]),
        "affine": nifti_affine(twixObj),
        "acs": None,
    }
    # Add sequence information
    for key in ["alTR", "alTE", "alTD", "alTI", "adFlipAngleDegree"]:
        # get a list of all sequences times in the sequence
        vals = twixObj.search_header_for_val("Phoenix", (f"{key}",))
        nice_key = key[2:]  # strip prefix "al /ad"
        if len(vals) == 1:
            hdr[nice_key] = vals[0]
        elif len(vals) > 0:
            # the first element found is the length of the list, we discard it.
            if vals[0] == len(vals[1:]):
                vals = vals[1:]
            hdr[nice_key] = vals
        # don't populate if not found.

    if "refscan" in twixObj.keys():
        twixObj.refscan.squeeze = True
        acs = twixObj.refscan[""].astype(np.complex64)
        hdr["acs"] = acs.swapaxes(0, 1)


    # from read_abrgrad_rawdat in mrinufft.io.nsp
    # we only want specific header for now.
    hdr["type"] = "ARBGRAD_GRE"
    hdr["shifts"] = ()
    for s in [6, 7, 8]:
        shift = twixObj.search_header_for_val(
            "Phoenix", ("sWiPMemBlock", "adFree", str(s))
        )
        hdr["shifts"] += (0,) if shift == [] else (shift[0],)
    hdr["oversampling_factor"] = twixObj.search_header_for_val(
        "Phoenix", ("sWiPMemBlock", "alFree", "4")
    )[0]
    hdr["trajectory_name"] = twixObj.search_header_for_val(
        "Phoenix", ("sWipMemBlock", "tFree")
    )[0][1:-1]
    if hdr["n_contrasts"] > 1:
        hdr["turboFactor"] = twixObj.search_header_for_val(
            "Phoenix", ("sFastImaging", "lTurboFactor")
        )[0]
        hdr["type"] = "ARBGRAD_MP2RAGE"
    return hdr, twixObj




def _hdr_to_sim_config(hdr:dict[str, Any]) -> SimConfig:
    """Convert header information to SimConfig."""
    
    try:
        seq=GreConfig(TR=hdr["TR"][-1], TE=hdr["TE"][-1], FA=hdr["FlipAngleDegree"][-1])
    except KeyError:
        log.warning("TR, TE, FA not found in header, using default GreConfig(50,30,18)")
        seq=GreConfig(50,30,18)
    sim_conf = SimConfig(
        fov=FOVConfig.from_affine(hdr["affine"], size=hdr["FOV"]),
        hardware=HardwareConfig(),
        seq=seq,
        max_sim_time=seq.TR * hdr["num_shots"] / 1000 
    )
    return sim_conf

class SiemensDataLoader(MRDLoader):
    
    def __init__(self, dat_file, bin_file, smaps_file, n_shot_per_frames=20, start_idx=0, is_accelerated=False,  **traj_args):
        #super().__init__()
        # TODO get the num_slices from the data itself ? 
        self._bin_file = bin_file
        self._smaps_file = smaps_file
        self._dat_file = dat_file
        self._level = 0
        self.start_idx = start_idx
        self.is_accelerated = is_accelerated
        

        self.n_shot_per_frames = n_shot_per_frames

        self.hdr, self.twixObj = parse_hdr_data(self._dat_file)
        self.traj, params = read_trajectory(self._bin_file, **traj_args)
        # merge params into hdr
        self.hdr = self.hdr | params
        print("init done")
        print(self.hdr)
        # assert self.hdr["n_adc_samples"] == self.hdr["num_samples_per_shot"]
        # assert self.hdr["n_shots"] == self.traj.shape[0]
    # --- Properties --- #

    
    @property
    def shape(self):
        # to be retrieved from header or smaps shape 
        return self.hdr["img_size"]

    @property
    def n_coils(self):
        return self.hdr["n_coils"]
    
    @property
    def n_frames(self):
        if self.is_accelerated:
        # to be retrieved from header a
            return self.hdr["n_shots"] // self.n_shot_per_frames
        else:
            return self.hdr['n_reps']

    @property
    def n_acquisitions(self):
        return self.hdr["n_shots"]

    @property
    def n_samples(self):
        return self.hdr["num_adc_samples"]
    

    @property
    def slice_2d(self) -> bool:
        return bool(np.asarray(self.hdr["img_size"])[-1] == 1)



    # --- Getters --- #

    def get_sim_conf(self) -> SimConfig:
        """Get the simulation configuration from the header."""
    
        return _hdr_to_sim_config(self.hdr)
        
    
    def get_smaps(self, resample=False) -> np.ndarray:
        """Get the smaps from the file."""
        with open(self._smaps_file, "rb") as f:
            smaps = np.load(f)
            smaps = np.ascontiguousarray(smaps[:, ::-1,::-1])
            return smaps
        # TODO add resampling support ?
        
        

    def get_kspace_frame(self, frame_idx: int, shot_dim: bool = False, traj_2d: bool = False) -> tuple[np.ndarray, np.ndarray]:
        # extract the data from the rawdata file 
        if self.is_accelerated:
            shot_idx = (self.start_idx + frame_idx) * self.n_shot_per_frames
            
            # data is nsamples, n_coils, n_shot. 
            raw_ksp = self.twixObj.image[:,:,shot_idx:shot_idx+self.n_shot_per_frames].squeeze()
            raw_ksp = np.moveaxis(raw_ksp, (0,1,2), (2,0,1))

            traj = self.traj[shot_idx:shot_idx+self.n_shot_per_frames]
        else:
            shot_idx = self.start_idx + frame_idx
            raw_ksp = self.twixObj.image[:,:,:,:,:,:,:,:,shot_idx].squeeze()
            raw_ksp = np.moveaxis(raw_ksp, (0,1,2), (2,0,1))
            traj = self.traj

        shifts = tuple(x / 1.5 for x in self.hdr["shifts"]) 
        raw_ksp_corrected = np.zeros_like(raw_ksp, dtype=np.complex64)
        for i in range(traj.shape[0]):
            raw_ksp_corrected[:, i, :] = add_phase_to_kspace_with_shifts(raw_ksp[:, i, :], traj[i], shifts)
        return traj, raw_ksp_corrected.reshape(raw_ksp_corrected.shape[0], -1) # flatten the shot and samples dimensions.



        
        # apply kspace shifts on the fly
        # return kspace and traj for the given frame index
        ...
    # --- Context manager --- #
    def __enter__(self):
        return self

    def exit__(self, exc_type, exc_value, traceback):
        pass


if __name__ == "__main__":
    # simple test
    dat_file = "/volatile/Caini/meas_MID00290_FID06797_bold_spiral_gre_tra_iso3mm_R2.dat"
    bin_file = "/volatile/Caini/stackofspiral_dynamic_r2_proj_smax0.2_192x192x120_240s.bin"
    smaps_file = "/path/to/smaps.npy"
    traj_args = {
        "dwell_time": 0.002,
        "raster_time": 0.01,
    }
    loader = SiemensDataLoader(dat_file, bin_file, smaps_file, n_shot_per_frames=20, **traj_args)
    sim_conf = loader.get_sim_conf()
    print(sim_conf)
    print(loader.hdr)
    print(loader.get_kspace_frame(0))
