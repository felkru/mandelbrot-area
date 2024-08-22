#!/usr/bin/env -S submit -M 2000 -m 2000 -f python -u

# This script is based on a Jupyter notebook provided by Jim Pivarski
# You can find it on GitHub: https://github.com/ErUM-Data-Hub/Challenges/blob/computing_challenge/computing/challenge.ipynb
# The markdown cells have been converted to raw comments
# and some of the LaTeX syntax has been removed for readability

import jax.numpy as jnp
import jax
from functools import partial

from utils import (
    combine_uncertainties,
    confidence_interval,
    wald_uncertainty,
)

# CONFIG
NUM_TILES_1D = 10
SAMPLES_IN_BATCH = (
    2  # Sample SAMPLES_IN_BATCH more points until uncert_target is reached
)
CONFIDENCE_LEVEL = 0.05

# Knill limits
xmin_knill, xmax_knill = -2, 1
ymin_knill, ymax_knill = -3 / 2, 3 / 2

width = 3 / NUM_TILES_1D
height = 3 / NUM_TILES_1D

rng_key = jax.random.key(0)


@jax.jit
def is_in_mandelbrot(x, y):
    c = jnp.complex64(x) + jnp.complex64(y) * jnp.complex64(1j)
    z_hare = z_tortoise = jnp.complex64(0)

    def body_fun(val):
        z_hare, z_tortoise, c = val
        jax.debug.print(
            "is_in_mandelbrot starts: z_hare = {z_h}, z_tortoise = {z_t}, c = {c}",
            z_h=z_hare,
            z_t=z_tortoise,
            c=c,
        )
        z_hare = z_hare * z_hare + c
        z_hare = z_hare * z_hare + c
        z_tortoise = z_tortoise * z_tortoise + c
        return [z_hare, z_tortoise, c]

    def cond_fun(val):
        not_converge = val[0] != val[1]
        not_diverge = (val[0].real ** 2 + val[0].imag ** 2) < 4
        # jax.debug.print(
        #     "not_converge = {not_c}, not_diverge = {not_d}",
        #     not_c=not_converge,
        #     not_d=not_diverge,
        # )
        return not_converge & not_diverge

    return jax.lax.while_loop(
        cond_fun,
        body_fun,
        body_fun([z_hare, z_tortoise, c]),
    )


@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def count_mandelbrot(rng_key, num_samples, xmin, width, ymin, height):
    """Draw num_samples random numbers uniformly between (xmin, xmin+width)
    and (ymin, ymin+height).
    Raise `out` by one if the number is part of the Mandelbrot set.
    """
    out = jnp.int32(0)
    x_norm = jax.random.uniform(rng_key, shape=(num_samples,))
    y_norm = jax.random.uniform(rng_key, shape=(num_samples,))

    x = xmin + (x_norm * width)
    y = ymin + (y_norm * height)
    z_h, z_t, c = jax.vmap(is_in_mandelbrot)(x, y)
    jax.debug.print("is_in_mandelbrot finished")

    result = (z_h.real**2 + z_h.imag**2) < 4

    out = jnp.sum(result)

    return out


@jax.jit
def xmin(j):
    """xmin of tile in column j"""
    return -2 + width * j


@jax.jit
def ymin(i):
    """ymin of tile in row i"""
    return -3 / 2 + height * i


@partial(jax.jit, static_argnames=["uncert_target"])
def compute_until(rng_key, uncert_target, i, j):
    """Compute area of each tile until uncert_target is reached.
    The uncertainty is calculate with the Wald approximation in each tile.
    """
    uncert = jnp.inf
    denom = 0
    numer = 0
    init_val = [denom, numer, uncert]

    def cond_fun(val):
        return val[2] > uncert_target

    def body_fun(val):
        jax.debug.print("compute_until starts")
        val[0] += SAMPLES_IN_BATCH
        val[1] += count_mandelbrot(
            rng_key, SAMPLES_IN_BATCH, xmin(j), width, ymin(i), height
        )
        jax.debug.print("count_mandelbrot finished")
        val[2] = wald_uncertainty(val[1], val[0]) * width * height
        jax.debug.print("wald_uncertainty finished")
        return val

    val = jax.lax.while_loop(
        cond_fun,
        body_fun,
        init_val,
    )
    jax.debug.print("loop finished")

    return val


denom, numer, uncert = jax.vmap(
    jax.vmap(compute_until, in_axes=(None, None, None, 0)),
    in_axes=(None, None, 0, None),
)(rng_key, 1e-5, jnp.arange(NUM_TILES_1D), jnp.arange(NUM_TILES_1D))

final_value = (jnp.sum((numer / denom)) * width * height).item()
jax.debug.print(f"\tThe total area of all tiles is {final_value}")

confidence_interval_low, confidence_interval_high = confidence_interval(
    CONFIDENCE_LEVEL, numer, denom, width * height
)

final_uncertainty = combine_uncertainties(
    confidence_interval_low, confidence_interval_high, denom
)
jax.debug.print(f"\tThe uncertainty on the total area is {final_uncertainty}\n")
