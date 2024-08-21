import jax.numpy as jnp
from jax import jit, random
import numpy as np
from functools import partial

# CONFIG
NUM_TILES_1D = 100
SAMPLES_IN_BATCH = 100  # Sample SAMPLES_IN_BATCH more points until uncert_target is reached
CONFIDENCE_LEVEL = 0.05


@jit
def is_in_mandelbrot(x, y):
    """Tortoise and Hare approach to check if point (x, y) is in Mandelbrot set."""
    c = jnp.complex64(x) + jnp.complex64(y) * jnp.complex64(1j)
    z_hare = z_tortoise = jnp.complex64(0)  # tortoise and hare start at the same point
    while True:
        z_hare = z_hare * z_hare + c
        z_hare = z_hare * z_hare + c  # hare does one step more to get ahead of the tortoise
        z_tortoise = z_tortoise * z_tortoise + c  # tortoise is one step behind
        if z_hare == z_tortoise:
            return True  # orbiting or converging to zero
        if z_hare.real ** 2 + z_hare.imag ** 2 > 4:
            return False  # diverging to infinity


@partial(jit, static_argnums=(1, 2, 3, 4))
def count_mandelbrot(rng_key, num_samples, xmin, width, ymin, height):
    """Draw num_samples random numbers uniformly between (xmin, xmin + width)
    and (ymin, ymin + height).
    Raise `out` by one if the number is part of the Mandelbrot set.
    """
    out = 0
    keys = random.split(rng_key, num_samples)
    x_norm = random.uniform(keys, shape=(num_samples,), minval=0, maxval=1)
    y_norm = random.uniform(keys, shape=(num_samples,), minval=0, maxval=1)

    for i in range(num_samples):
        x = xmin + (x_norm[i] * width)
        y = ymin + (y_norm[i] * height)
        out += is_in_mandelbrot(x, y)

    return out


# Knill limits
xmin_value, xmax_value = -2, 1
ymin_value, ymax_value = -3 / 2, 3 / 2

width = 3 / NUM_TILES_1D
height = 3 / NUM_TILES_1D


@jit
def compute_xmin(j):
    """xmin of tile in column j"""
    return -2 + width * j


@jit
def compute_ymin(i):
    """ymin of tile in row i"""
    return -3 / 2 + height * i


rng_key = random.PRNGKey(0)  # JAX's random key
rng_keys = random.split(rng_key, NUM_TILES_1D * NUM_TILES_1D)


@jit
def compute_until(rng_keys, numer, denom, uncert, uncert_target):
    """Compute area of each tile until uncert_target is reached.
    The uncertainty is calculated with the Wald approximation in each tile.
    """

    def body_fn(i, state):
        numer, denom, uncert = state

        def body_j(j, inner_state):
            numer, denom, uncert = inner_state
            rng = rng_keys[NUM_TILES_1D * i + j]

            def cond_fn(carry):
                _, _, uncert_value = carry
                return uncert_value > uncert_target

            def loop_body(carry):
                numer_val, denom_val, uncert_value = carry
                denom_val += SAMPLES_IN_BATCH
                numer_val += count_mandelbrot(
                    rng, SAMPLES_IN_BATCH, compute_xmin(j), width, compute_ymin(i), height
                )
                uncert_value = wald_uncertainty(numer_val, denom_val) * width * height
                return numer_val, denom_val, uncert_value

            numer[i, j], denom[i, j], uncert[i, j] = jax.lax.while_loop(
                cond_fn, loop_body, (numer[i, j], denom[i, j], uncert[i, j])
            )

            return numer, denom, uncert

        numer, denom, uncert = jax.lax.fori_loop(0, NUM_TILES_1D, body_j, (numer, denom, uncert))
        return numer, denom, uncert

    numer, denom, uncert = jax.lax.fori_loop(0, NUM_TILES_1D, body_fn, (numer, denom, uncert))
    return numer, denom, uncert


numer = jnp.zeros((NUM_TILES_1D, NUM_TILES_1D), dtype=jnp.int64)
denom = jnp.zeros((NUM_TILES_1D, NUM_TILES_1D), dtype=jnp.int64)
uncert = jnp.zeros((NUM_TILES_1D, NUM_TILES_1D), dtype=jnp.float64)

numer, denom, uncert = compute_until(rng_keys, numer, denom, uncert, 1e-5)

final_value = (jnp.sum((numer / denom)) * width * height).item()
print(f"\tThe total area of all tiles is {final_value}")

confidence_interval_low, confidence_interval_high = confidence_interval(
    CONFIDENCE_LEVEL, numer, denom, width * height
)

final_uncertainty = combine_uncertainties(
    confidence_interval_low, confidence_interval_high, denom
)
print(f"\tThe uncertainty on the total area is {final_uncertainty}\n")
