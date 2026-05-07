# %%
"""
Single anatomical EPI with SNAKE-fMRI, with off-resonance effects
=================================================================
"""

# %%

# Imports
import numpy as np

from snake.core.phantom import Phantom
from snake.core.sampling import EPI3dAcquisitionSampler
from snake.core.simulation import GreConfig, SimConfig, default_hardware
from snake.toolkit.plotting import axis3dcut


# %%
# Setting up the base simulation Config.  This configuration holds all key
# parameters for the simulation, describing the scanner parameters.

sim_conf = SimConfig(
    max_sim_time=6,
    seq=GreConfig(TR=50, TE=30, FA=3),
    hardware=default_hardware,
)
sim_conf.hardware.n_coils = 8
sim_conf.fov.res_mm = (3, 3, 3)
sim_conf

# %%
# Creating the base Phantom
# -------------------------
# Here we use the phantom build from in-vivo data 

phantom = Phantom.from_nifti(


# %%
# Setting up Acquisition Pattern and Initializing Result file.
# ------------------------------------------------------------

# The next piece of simulation is the acquisition trajectory.
# Here nothing fancy, we are using a EPI (fully sampled), that samples a 3D
# k-space (this akin to the 3D EPI sequence of XXXX)

sampler = EPI3dAcquisitionSampler(accelz=1, acsz=0.1, orderz="top-down")

# %%
# Acquisition with Cartesian Engine
# ---------------------------------
#
# The generated file ``example_EPI.mrd`` does not contains any k-space data for
# now, only the sampling trajectory. let's put some in. In order to do so, we
# need to setup the **acquisition engine** that models the MR physics, and get
# sampled at the specified k-space trajectory.
#
# SNAKE comes with two models for the MR Physics:
#
# - model="simple" :: Each k-space shot acquires a constant signal, which is the
#   image contrast at TE.
# - model="T2s" :: Each k-space shot is degraded by the T2* decay induced by
#   each tissue.

# Here we will use the "simple" model, which is faster.
#
# SNAKE's Engine are capable of simulating the data in parallel, by distributing
# the shots to be acquired to a set of processes. To do so , we need to specify
# the number of jobs that will run in parallel, as well as the size of a job.
# Setting the job size and the number of jobs can have a great impact on total
# runtime and memory consumption.
#
# Here, we have a single frame to acquire with 60 frames (one EPI per slice), so
# a single worker will do.

from snake.core.engine import EPIAcquisitionEngine


engine = EPIAcquisitionEngine(model="simple")

engine(
    "example_EPI.mrd",
    sampler=sampler,
    phantom=phantom,
    sim_conf=sim_conf,
    worker_chunk_size=16,
    n_workers=4,
)

# %%
# Simple reconstruction
# ---------------------
#
# Getting k-space data is nice, but
# SNAKE also provides rudimentary reconstruction tools to get images (and check
# that we didn't mess up the acquisition process).
# This is available in the companion package ``snake.toolkit``.
#
# Loading the ``.mrd`` file to retrieve all information can be done using the
# ``ismrmd`` python package, but SNAKE provides convenient dataloaders, which are
# more efficient, and take cares of managing underlying files access. As we are
# showcasing the API, we will do things manually here, and use only core SNAKE.

from snake.mrd_utils import CartesianFrameDataLoader

with CartesianFrameDataLoader("example_EPI.mrd") as data_loader:
    mask, kspace_data = data_loader.get_kspace_frame(0)


# %%
with CartesianFrameDataLoader("example_EPI.mrd") as data_loader:
    phantom = data_loader.get_phantom()

# %%
sim_conf.fov.affine

# %%
# Reconstructing a Single Frame of fully sampled EPI boils down to performing a 3D IFFT:

from scipy.fft import fftshift, ifftn, ifftshift

axes = (-3, -2, -1)
image_data = ifftshift(
    ifftn(fftshift(kspace_data, axes=axes), axes=axes, norm="ortho"), axes=axes
)

# Take the square root sum of squares to get the magnitude image (SSOS)
image_data = np.sqrt(np.sum(np.abs(image_data) ** 2, axis=0))

# %% plotting the result

import matplotlib.pyplot as plt


fig, ax = plt.subplots()

axis3dcut(image_data.squeeze().T, None, None, cbar=False, cuts=(0.5, 0.5, 0.5), ax=ax)
plt.show()

# %%
