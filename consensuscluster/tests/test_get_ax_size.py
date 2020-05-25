"""Contains unit tests for the function _get_ax_size.

There are a lot of moving parts involved in writing these tests, so I
felt it was best to encapsulate them all in their own file. The reason
there's so much going on here is that we want to test a whole bunch of
combinations of:
* Various Figure-sizes
* Various DPI values
* Various methods of Figure creation
* Various patterns of Axes creation on the Figure

The goal is to test that _get_ax_size returns an answer that is within
a reasonable margin of the answer you'd get by hand. Because I just
grabbed this function off of stackoverflow without any evidence that it
was actually correct, it's important to really test the bejesus out of
it.
"""

from math import floor, ceil
import pytest
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from matplotlib.testing import decorators

from consensuscluster.plotutils import _get_ax_size

# These are the maximum fractions of error we're ok with.
# There are different fractions for the lower-bound and upper-bound
# because for _get_ax_size, we would prefer to overestimate rather
# than underestimate.
MAX_ERROR_LOWER = .06
MAX_ERROR_UPPER = .1

figsizes = [
    (1, 1),
    (3, 3),
    (4, 4),
    (4, 9),
    (.87, .4445),
    (5.829, 1)
]

dpis = [100, 43.793]

figure_creation_funcs = [
    lambda figsize, dpi: Figure(figsize=figsize, dpi=dpi),
    lambda figsize, dpi: plt.figure(figsize=figsize, dpi=dpi)
]


def _check_answer(true_width_pix, true_height_pix,
                  approx_width_pix, approx_height_pix):
    """Helper function for testing _get_ax_size.

    Asserts that the answer found by _get_ax_size is within the
    acceptable margin of error of the true answer (or at least,
    whatever we're considering the true answer to be).

    :param true_width_pix: True width of the Axes, in pixels.
    :param true_height_pix: True height of the Axes, in pixels.
    :param approx_width_pix: The width approximation returned by
    _get_ax_size.
    :param approx_height_pix: The height approximation returned by
    _get_ax_size.
    :return: nothing.
    """

    # Here I round the bounds down/up to the nearest pixel depending on
    # whether it's a lower or upper bound. The main reason for this is
    # to make the tests for really small widths/heights more lax, bc
    # often the approximate will be within a pixel of the real answer
    # but the test will still fail.
    # -Vicram
    width_lower_bound = true_width_pix - MAX_ERROR_LOWER * true_width_pix
    width_lower_bound = floor(width_lower_bound)

    width_upper_bound = true_width_pix + MAX_ERROR_UPPER * true_width_pix
    width_upper_bound = ceil(width_upper_bound)

    height_lower_bound = true_height_pix - MAX_ERROR_LOWER * true_height_pix
    height_lower_bound = floor(height_lower_bound)

    height_upper_bound = true_height_pix + MAX_ERROR_UPPER * true_height_pix
    height_upper_bound = ceil(height_upper_bound)

    assert width_lower_bound <= approx_width_pix <= width_upper_bound
    assert height_lower_bound <= approx_height_pix <= height_upper_bound


def _check_answer_subplots(fig, axarr, rows, cols,
                           total_width_pix, total_height_pix):
    """Check _get_ax_size on every Axes in an array of Axes (subplots).

    This function will compute the "correct" width/height pixels using
    the number of rows/cols and then check the output of _get_ax_size
    against these for EACH Axes in the axarr.

    :param fig: Parent Figure containing the subplots.
    :param axarr: Array of Axes containing equal-sized subplots.
    :param rows: Number of rows of subplots in the full Figure.
    :param cols: Number of columns of subplots in the full Figure.
    :param total_width_pix: Total width (in pixels) of the full Figure.
    :param total_height_pix: Total height (in pixels) of the full
    Figure.
    :return: nothing.
    """
    correct_width_sub = total_width_pix / cols  # "True" width, in pixels
    correct_height_sub = total_height_pix / rows
    for i in range(rows):
        for j in range(cols):
            ax_sub = axarr[i, j]
            (approx_width_sub, approx_height_sub) = _get_ax_size(
                ax_sub,
                fig
            )
            _check_answer(correct_width_sub, correct_height_sub,
                          approx_width_sub, approx_height_sub)


@pytest.mark.parametrize('figsize', figsizes)
@pytest.mark.parametrize('dpi', dpis)
@pytest.mark.parametrize('figfunc', figure_creation_funcs)
@decorators.cleanup
def test_ax_and_axarr(figsize, dpi, figfunc):
    """Test creating a single Axes then an Axes array on the same fig.

    :return: nothing.
    """
    (width, height) = figsize  # True values, in inches

    # True values, in pixels
    width_pix = width * dpi
    height_pix = height * dpi

    fig = figfunc(figsize, dpi)
    ax = fig.gca()
    # ax should cover the entire figure.
    (approx_width, approx_height) = _get_ax_size(ax, fig)
    _check_answer(width_pix, height_pix,
                  approx_width, approx_height)

    # Second, create a subplot on that same Figure
    axarr = fig.subplots(5, 3)
    _check_answer_subplots(fig, axarr, 5, 3,
                           width_pix, height_pix)
