import jax.numpy as jnp
from jax.scipy.stats import beta
import jax
import matplotlib.pyplot as plt


def plot_pixels(pixels, figsize=(7, 7), dpi=300, extend=[-2, 1, -3 / 2, 3 / 2]):
    # Ensure the pixels are a JAX array
    pixels = jnp.array(pixels)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, layout="constrained")
    p = ax.imshow(pixels, extent=extend)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    return fig, ax, p


def confidence_interval(confidence_level, numerator, denominator, area):
    """Calculate confidence interval based on Clopper-Pearson using JAX.
    `beta.ppf` is the Percent Point function of the Beta distribution.
    """

    # Calculate the lower bound of the confidence interval
    low = (
        jnp.nan_to_num(
            beta.ppf(confidence_level / 2, numerator, denominator - numerator + 1),
            nan=0,
        )
        * area
    )

    # Calculate the upper bound of the confidence interval
    high = (
        jnp.nan_to_num(
            beta.ppf(1 - confidence_level / 2, numerator + 1, denominator - numerator),
            nan=1,
        )
        * area
    )

    # Ensure that the low and high bounds are valid
    low = jnp.nan_to_num(jnp.asarray(low), nan=0)
    high = jnp.nan_to_num(jnp.asarray(high), nan=area)

    return low, high


@jax.jit
def wald_uncertainty(numer, denom):
    """Wald approximation on the uncertainty of the tile using JAX."""

    # Handle edge cases
    def true_fun(denom):
        numer = 1
        denom += 1
        return numer / denom

    def false_true_fun():
        return numer / (denom + 1)

    def false_false_fun():
        return numer / denom

    def false_fun(denom):
        return jax.lax.cond(
            numer == denom,
            false_true_fun,
            false_false_fun,
        )

    frac = jax.lax.cond(numer == 0, true_fun, false_fun, denom)

    # Return the Wald uncertainty
    return jnp.sqrt(frac * (1 - frac) / denom)


def combine_uncertainties(
    confidence_interval_low, confidence_interval_high, denominator
):
    """
    Combine uncertainties using the formula from the stratified sampling method.
    """
    # Calculate the final uncertainty
    final_uncertainty = (
        jnp.sum(confidence_interval_high - confidence_interval_low)
        / jnp.sqrt(4 * jnp.sum(denominator))
    ).item()

    return final_uncertainty
