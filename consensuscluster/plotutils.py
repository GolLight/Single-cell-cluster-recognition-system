"""Functions to help create plots.
"""
import numpy as np
from matplotlib.colors import Normalize
from scipy.sparse import issparse
from skimage.transform import resize  # TODO get rid of skimage dependency

from .misc import IS_TEST
from .misc import printif
from .misc import (DEBUGLVL, USERLVL)

NOP_NORM = Normalize(0.0, 1.0)
"""Instance of Normalize which is a no-op.

Normally, to assign colors to values in the consensus matrix,
matplotlib would rescale (normalize) the values in the matrix to be in
[0,1], but that's not the desired behavior. We would like the values to
simply be treated as values in [0,1] without modification, even if the
actual minimum value is higher than 0 or the actual max is less than 1.
The way to ensure that no normalization will be done is to provide our
own Normalizer and make it a no-op. 
"""


def _get_ax_size(ax, fig):
    """Find the size of an Axes, in pixels.

    Given an Axes and its parent Figure, this function will approximate
    the width and height of the Axes and return them in units of
    pixels. The code for this function was taken from `this
    stackoverflow post
    <https://stackoverflow.com/questions/19306510/determine-matplotlib-axis-size-in-pixels>`_.

    Parameters
    ----------
    ax : matplotlib Axes object
        The Axes for which you want to compute the dimensions. This may
        be part of a subplot.

    fig : matplotlib Figure object
        The Figure which contains ax.

    Returns
    -------
    width : int
        The approximate width of ax, rounded to the nearest pixel.

    height : int
        The approximate height of ax, rounded to the nearest pixel.
    """
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    width *= fig.dpi
    height *= fig.dpi
    # Note: There are some magic numbers here. Here's the explanation:
    # in my testing, I found that the approximated width and height
    # were consistently less than the true values. This is probably
    # because the actual Axes is shrunk a bit to make room for the
    # x- and y-axis.
    # I played around with the precise ratios until I was able to find
    # numbers such that all the tests would pass.
    # -Vicram
    return (int(width * 1.411), int(height * 1.365))


def plot_consensus_heatmap(ordered_cmat, ax, fig, cmap, downsample, verbose):
    """Plot the given consensus matrix as a heatmap.

    This function plots the consensus heatmap onto the given Axes. The
    consensus matrix must have already been reordered to group samples
    in each cluster together. This is required to get the "block
    matrix" look.

    Normally, you won't need to do anything with this function's return
    value. This function is written with a style advocated in `the
    matplotlib Usage FAQ
    <https://matplotlib.org/faq/usage_faq.html#coding-styles>`_.

    To actually plot the heatmap, this function will call
    matplotlib.axes.Axes.imshow. imshow has some potential memory
    issues; see `this stackoverflow post
    <https://stackoverflow.com/questions/9525706/excessive-memory-usage-in-matplotlib-imshow>`_.
    This is likely caused by imshow storing the entire matrix and doing
    computations on it at plotting-time. To deal with this, we can
    downsample the consensus matrix before passing it to imshow. This
    just involves creating a smaller matrix that would look about the
    same if plot, then plotting the smaller matrix.

    Parameters
    ----------
    ordered_cmat : ndarray
        This is the consensus matrix to plot. Must be a symmetric
        2-dimensional ndarray with values between 0 and 1. The values
        should have been reordered to group samples in the same cluster
        together.

    ax : matplotlib Axes object
        The Axes onto which the heatmap will be drawn.

    fig : matplotlib Figure object
        Must be the Figure which contains ax. It will not be mutated;
        this function will only be reading some of its attributes.

    cmap : str or Colormap
        This param determines the colormap which will be used to draw
        the heatmap. It can be anything which can be passed to the
        'cmap' param of imshow. Usually this will be a string which
        corresponds to the name of a colormap. For names of matplotlib
        colormaps, as well as recommendations for their usage, see
        `this tutorial
        <https://matplotlib.org/tutorials/colors/colormaps.html>`_.

    downsample: boolean
        Determines whether the consensus matrix will be downsampled
        before it is passed to imshow.

    verbose: non-negative int
        Verbosity level of print statements. If this is 0, no output
        will be produced. If >= 1, some print statements will trigger.

    Returns
    -------
    matplotlib AxesImage object
        This is the return value from calling imshow.
    """
    printif(
        verbose >= DEBUGLVL,
        'Entering plot_consensus_heatmap'
    )
    assert isinstance(verbose, int)
    if IS_TEST:
        printif(
            verbose >= DEBUGLVL,
            'IS_TEST is true, so now validating ordered_cmat'
        )
        # We want to check that ordered_cmat is:
        # * an ndarray
        # * 2D
        # * in [0,1]
        # * symmetric
        assert isinstance(ordered_cmat, np.ndarray)
        assert not issparse(ordered_cmat)
        assert ordered_cmat.ndim == 2
        assert np.all(
            np.logical_and(
                ordered_cmat >= 0.0,
                ordered_cmat <= 1.0
            )
        )
        assert np.array_equal(ordered_cmat, ordered_cmat.T)  # check symmetric

    # Next, deal with downsampling and interpolation.
    (width, height) = _get_ax_size(ax, fig)
    assert width == height  # The Axes must be square.
    if not downsample:
        mat_to_plot = ordered_cmat
    else:
        # Here, we will downsample the consensus matrix before passing
        # it to imshow. This is to reduce imshow's memory overhead.
        # However, plot quality is still the top priority, so we need
        # to ensure that downsampling won't meaningfully change the
        # plot which is produced. To solve this problem, we make the
        # following assumption: if the downsampled image still has more
        # pixels than the output plot would, then the plot will
        # probably end up being pretty similar, as in either case,
        # imshow would need to do its own additional downsampling in
        # order to render the image.
        # (Note: we are abstractly treating each value in the matrix
        # as a pixel, although depending on the resolution and size of
        # the output each value could end up corresponding to multiple
        # pixels or less than a pixel.)
        printif(verbose >= DEBUGLVL,
                'Now computing dimensions to downsample to')
        # To leave some room for error, we only downsample down to
        # four times the number of pixels in the Axes.
        n_samples = ordered_cmat.shape[0]
        target_len = 2 * width
        if n_samples <= target_len:
            mat_to_plot = ordered_cmat
        else:
            mat_to_plot = resize(ordered_cmat, (target_len, target_len))  # TODO choose mode
            assert mat_to_plot.shape == (target_len, target_len)
    # The imshow function has an 'interpolation' parameter which
    # determines the algorithm that will be used for downsampling or
    # upsampling when rendering the image. This is IN ADDITION TO
    # our own downsampling; we're doing downsampling to improve
    # imshow's performance, but imshow will also apply an interpolation
    # function itself.
    # See here for some documentation on interpolation methods:
    # https://matplotlib.org/gallery/images_contours_and_fields/interpolation_methods.html
    # Among other things, this page states that:
    # * If interpolation = "none" then it will default to
    #   interpolation = "nearest", except for the Agg, ps, and pdf
    #   backends.
    # * For Agg, ps, and pdf, it is best to use "none" when the matrix
    #   is bigger than the Axes, and it's best to use "nearest" when
    #   the matrix is smaller than the Axes.
    # Therefore, by using "none" when the matrix is bigger and
    # "nearest" when it is smaller, we will always get the desired
    # behavior.
    side_len = mat_to_plot.shape[0]
    if side_len >= width:
        interpolation = 'none'
    else:
        interpolation = 'nearest'

    out = ax.imshow(
        mat_to_plot,
        cmap=cmap,
        norm=NOP_NORM,
        aspect='equal',
        interpolation=interpolation,
        origin='upper'
    )  # TODO resample param
    printif(verbose >= DEBUGLVL, 'Now exiting plot_consensus_heatmap')
    return out
