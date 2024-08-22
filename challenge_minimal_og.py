#!/usr/bin/env -S submit -M 2000 -m 2000 -f python -u

# This script is based on a Jupyter notebook provided by Jim Pivarski
# You can find it on GitHub: https://github.com/ErUM-Data-Hub/Challenges/blob/computing_challenge/computing/challenge.ipyjax
# The markdown cells have been converted to raw comments
# and some of the LaTeX syntax has been removed for readability

import warnings
from functools import partial

import jax
import jax.numpy as jnp

# ignore deprecation warnings from numba for now
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

from utils import (
    combine_uncertainties,
    confidence_interval,
    wald_uncertainty,
)

warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)

jax.config.update("jax_enable_x64", True)


# CONFIG
NUM_TILES_1D = 10
SAMPLES_IN_BATCH = (
    100  # Sample SAMPLES_IN_BATCH more points until uncert_target is reached
)
CONFIDENCE_LEVEL = 0.05


def is_in_mandelbrot(x, y):
    """Toirtoise and Hare approach to check if point (x,y) is in Mandelbrot set."""
    c = jnp.complex64(x) + jnp.complex64(y) * jnp.complex64(1j)
    z_hare = z_tortoise = jnp.complex64(0)  # tortoise and hare start at same point
    while True:
        z_hare = z_hare * z_hare + c
        z_hare = (
            z_hare * z_hare + c
        )  # hare does one step more to get ahead of the tortoise
        z_tortoise = z_tortoise * z_tortoise + c  # tortoise is one step behind
        if z_hare == z_tortoise:
            return True  # orbiting or converging to zero
        if z_hare.real**2 + z_hare.imag**2 > 4:
            return False  # diverging to infinity


def diverges(z):
    return z.real**2 + z.imag**2 > 4


# @jax.jit
def mandelbrot_jax(x, y):
    c = x + y * 1j
    init_vals = (jnp.complex64(0), jnp.complex64(0), c)

    def cond_fun(val):
        diverge = (val[0].real ** 2 + val[0].imag ** 2) > 4
        not_converge = val[0] != val[1]
        # jax.debug.print(
        #     "val: {val}, diverge {x}, not_converge {y}",
        #     val=val,
        #     x=diverge,
        #     y=not_converge,
        # )
        return ~diverge & not_converge

    def body_fun(val):
        zhare, ztort, c = val

        zhare = zhare * zhare + c
        zhare = zhare * zhare + c

        ztort = ztort * ztort + c

        return (zhare, ztort, c)

    return jax.lax.while_loop(cond_fun, body_fun, body_fun(init_vals))


# @jax.jit
def count_mandelbrot(key, num_samples, xmin, width, ymin, height):
    """Draw num_samples random numbers uniformly between (xmin, xmin+width)
    and (ymin, ymin+height).
    Raise `out` by one if the number is part of the Mandelbrot set.
    """

    x_norm = jax.random.uniform(key, (num_samples,))
    y_norm = jax.random.uniform(key, (num_samples,))
    x = xmin + (x_norm * width)
    y = ymin + (y_norm * height)
    zhare, ztort, c = jax.vmap(mandelbrot_jax)(x, y)

    diverged = diverges(zhare)

    return jnp.sum(~diverged)


# Knill limits
xmin, xmax = -2, 1
ymin, ymax = -3 / 2, 3 / 2

width = 3 / NUM_TILES_1D
height = 3 / NUM_TILES_1D


@jax.jit
def xmin(j):
    """xmin of tile in column j"""
    return -2 + width * j


@jax.jit
def ymin(i):
    """ymin of tile in row i"""
    return -3 / 2 + height * i


def compute_until(subkeys, numer, denom, uncert, uncert_target):
    """Compute area of each tile until uncert_target is reached.
    The uncertainty is calculate with the Wald approximation in each tile.
    """
    for i in range(NUM_TILES_1D):
        for j in range(NUM_TILES_1D):
            key = subkeys[NUM_TILES_1D * i + j]

            uncert[i, j] = jnp.inf

            # Sample SAMPLES_IN_BATCH more points until uncert_target is reached
            while uncert[i, j] > uncert_target:
                denom[i, j] += SAMPLES_IN_BATCH
                numer[i, j] += count_mandelbrot(
                    key, SAMPLES_IN_BATCH, xmin(j), width, ymin(i), height
                )

                uncert[i, j] = (
                    wald_uncertainty(numer[i, j], denom[i, j]) * width * height
                )


# def run_jax_gpu():
#     y, x = jnp.ogrid[-2:1:100j, -1.5:0:100j]


@partial(jax.jit, static_argnames=("uncert_target"))
def compute_until_jax(key, uncert_target, i, j):
    uncert = jnp.inf
    denom = 0
    numer = 0
    init_val = [denom, numer, uncert]

    def cond_fun(val):
        return val[2] > uncert_target

    def body_fun(val):
        val[0] += SAMPLES_IN_BATCH
        jax.debug.print("before count_mandelbrot")
        val[1] += count_mandelbrot(
            key, SAMPLES_IN_BATCH, xmin(j), width, ymin(i), height
        )

        jax.debug.print("after count_mandelbrot")
        val[2] = wald_uncertainty(val[1], val[0]) * width * height
        jax.debug.print("val: {val}", val=val)
        return val

    val = jax.lax.while_loop(
        cond_fun,
        body_fun,
        init_val,
    )

    return val


compute_until_vmap = jax.vmap(
    jax.vmap(
        compute_until_jax,
        in_axes=(None, None, 0, None),
    ),
    in_axes=(None, None, None, 0),
)


numer = jnp.zeros((NUM_TILES_1D, NUM_TILES_1D), dtype=jnp.int64)
denom = jnp.zeros((NUM_TILES_1D, NUM_TILES_1D), dtype=jnp.int64)
uncert = jnp.zeros((NUM_TILES_1D, NUM_TILES_1D), dtype=jnp.float64)


key = jax.random.key(2)

print("RUNNING")

# denom_arr, numer_arr, unc_arr = compute_until_vmap(
#     key, 1e-5, jnp.arange(NUM_TILES_1D), jnp.arange(NUM_TILES_1D)
# )
# # compute_until(subkeys, numer, denom, uncert, 1e-5)

# final_value = (jnp.sum((numer / denom)) * width * height).item()
# print(f"\tThe total area of all tiles is {final_value}")

# confidence_interval_low, confidence_interval_high = confidence_interval(
#     CONFIDENCE_LEVEL, numer, denom, width * height
# )

# final_uncertainty = combine_uncertainties(
#     confidence_interval_low, confidence_interval_high, denom
# )
# print(f"\tThe uncertainty on the total area is {final_uncertainty}\n")

y, x = jnp.ogrid[-2:1:100j, -1.5:0:100j]
c = x + y * 1j
