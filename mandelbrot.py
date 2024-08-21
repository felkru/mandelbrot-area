import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numba as nb
import numba.cuda as cuda
import cupy as cp
# i know i shouldn't do that
from utils import *

@nb.njit
def is_in_mandelbrot(x, y):
    """Toirtoise and Hare approach to check if point (x,y) is in Mandelbrot set."""
    c = np.complex64(x) + np.complex64(y) * np.complex64(1j)
    z_hare = z_tortoise = np.complex64(0)  # tortoise and hare start at same point
    while True:
        z_hare = z_hare * z_hare + c
        z_hare = (
                z_hare * z_hare + c
        )  # hare does one step more to get ahead of the tortoise
        z_tortoise = z_tortoise * z_tortoise + c  # tortoise is one step behind
        if z_hare == z_tortoise:
            return True  # orbiting or converging to zero
        if z_hare.real ** 2 + z_hare.imag ** 2 > 4:
            return False  # diverging to infinity


@nb.njit
def count_mandelbrot(rng, num_samples, xmin, width, ymin, height):
    """Draw num_samples random numbers uniformly between (xmin, xmin+width)
    and (ymin, ymin+height).
    Raise `out` by one if the number is part of the Mandelbrot set.
    """
    out = np.int32(0)
    for x_norm, y_norm in rng.random((num_samples, 2), np.float32):
        x = xmin + (x_norm * width)
        y = ymin + (y_norm * height)
        out += is_in_mandelbrot(x, y)
    return out


@cuda.jit(device=True)
def is_in_mandelbrot_gpu(x, y):
    c = cp.complex64(x) + cp.complex64(y) * cp.complex64(1j)
    z_hare = z_tortoise = cp.complex64(0)
    while True:
        z_hare = z_hare * z_hare + c
        z_hare = z_hare * z_hare + c
        z_tortoise = z_tortoise * z_tortoise + c
        if z_hare == z_tortoise:
            return True
        if z_hare.real ** 2 + z_hare.imag ** 2 > 4:
            return False


@cuda.jit
def count_mandelbrot_gpu(rngs, num_samples, xmin, width, ymin, height, out):
    """GPU-accelerated Mandelbrot set count using CUDA."""
    i = cuda.grid(1)

    if i < num_samples:
        # Generate random numbers for each thread
        x_norm = rngs[i, 0]
        y_norm = rngs[i, 1]

        # Map to the Mandelbrot coordinates
        x = xmin + (x_norm * width)
        y = ymin + (y_norm * height)

        # Increment output if the point is in the Mandelbrot set
        if is_in_mandelbrot_gpu(x, y):
            cuda.atomic.add(out, 0, 1)


@nb.njit(parallel=True)
def compute_until(rngs, numer, denom, uncert, uncert_target):
    """Compute area of each tile until uncert_target is reached.
    The uncertainty is calculate with the Wald approximation in each tile.
    """
    for i in nb.prange(NUM_TILES_1D):
        for j in nb.prange(NUM_TILES_1D):
            rng = rngs[NUM_TILES_1D * i + j]

            uncert[i, j] = np.inf

            # Sample SAMPLES_IN_BATCH more points until uncert_target is reached
            while uncert[i, j] > uncert_target:
                denom[i, j] += SAMPLES_IN_BATCH
                numer[i, j] += count_mandelbrot_gpu(rng, SAMPLES_IN_BATCH, xmin(j), width, ymin(i), height)

                uncert[i, j] = (wald_uncertainty(numer[i, j], denom[i, j]) * width * height)


@nb.njit
def xmin(j):
    """xmin of tile in column j"""
    return -2 + width * j


@nb.njit
def ymin(i):
    """ymin of tile in row i"""
    return -3 / 2 + height * i


Text
cell < GONjAVUNotTC >
# %% [markdown]
Parameters

Code
cell < apEoVrA8LSse >
# %% [code]
rng = np.random.default_rng(seed=42)

NUM_TILES_1D = 100

width = 3 / NUM_TILES_1D
height = (3 / NUM_TILES_1D) / 2

rngs = rng.spawn(NUM_TILES_1D * NUM_TILES_1D)

SAMPLES_IN_BATCH = 100

numer = np.zeros((NUM_TILES_1D, NUM_TILES_1D), dtype=np.int64)
denom = np.zeros((NUM_TILES_1D, NUM_TILES_1D), dtype=np.int64)
uncert = np.zeros((NUM_TILES_1D, NUM_TILES_1D), dtype=np.float64)

# Define the parameters
num_samples = SAMPLES_IN_BATCH
xmin_val = -2
ymin_val = -3 / 2
width_val = 3
height_val = 3

# Use cupy to generate random numbers on the GPU
rngs_gpu = cp.random.random((num_samples, 2), dtype=cp.float32)

# Allocate output on GPU
out_gpu = cp.zeros(1, dtype=cp.int32)

# Define threads per block and number of blocks
threads_per_block = 32
blocks_per_grid = (num_samples + (threads_per_block - 1)) // threads_per_block

# Launch the kernel
count_mandelbrot_gpu[blocks_per_grid, threads_per_block](rngs_gpu, num_samples, xmin_val, width_val, ymin_val,
                                                         height_val, out_gpu)

# Copy result back to host
out_cpu = out_gpu.get()

print(f"Count of points in Mandelbrot set: {out_cpu[0]}")


# %% [code]
@nb.njit(parallel=True)
def compute_until(rngs, numer, denom, uncert, uncert_target):
    """Compute area of each tile until uncert_target is reached.
    The uncertainty is calculate with the Wald approximation in each tile.
    """
    for i in nb.prange(NUM_TILES_1D):
        for j in nb.prange(NUM_TILES_1D):
            rng = rngs[NUM_TILES_1D * i + j]

            uncert[i, j] = np.inf

            # Sample SAMPLES_IN_BATCH more points until uncert_target is reached
            while uncert[i, j] > uncert_target:
                denom[i, j] += SAMPLES_IN_BATCH
                numer[i, j] += count_mandelbrot_gpu(rng, SAMPLES_IN_BATCH, xmin(j), width, ymin(i), height)

                uncert[i, j] = (wald_uncertainty(numer[i, j], denom[i, j]) * width * height)


Text
cell < GYE1basUOvLj >
# %% [markdown]
The
total
area
of
all
tiles is 1.5060379066774943
The
uncertainty
on
the
total
area is 0.00013860334249145328


