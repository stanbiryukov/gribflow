import jax
import jax.numpy as jnp
import jax.scipy.ndimage as jndi
import numpy as np
from jax import jit
from PIL import Image

import cv2
from memoization import cached


def np_to_gray(ar, floor, ceil):
    # to uint8 0-255
    ar = 255 * (ar - floor) / (ceil - floor)
    # rgb
    a = np.array(Image.fromarray(np.uint8(ar)).convert("RGB"))
    # bgr
    im_bgr = cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
    # gray
    gray1 = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY)
    return gray1


def mm_scale(ar, floor, ceil):
    """
    Simple MinMax Scale. Rescale array to floor-ceil space.
        ie: convert gray scale array back to original space.
    """
    ar = (ar - np.nanmin(ar)) / (np.nanmax(ar) - np.nanmin(ar))
    ar = ((ceil - floor) * ar) + floor
    return ar


def masked_blend(x1, x2, weights):
    """
    Numpy NaN masked weighted average.
    """
    x1m = np.ma.MaskedArray(x1, mask=~np.isfinite(x1))
    x2m = np.ma.MaskedArray(x2, mask=~np.isfinite(x2))
    es = np.ma.average([x1m, x2m], weights=weights, axis=0).filled(np.nan)
    return es


@jit
def geometric_blend(x1, x2, weights):
    """
    geometric mean of two arrays where non-finites (ie log(0)) are replaced with arithmetic mean
    """
    x1m = jnp.log(x1)
    x2m = jnp.log(x2)
    es = jnp.average([x1m, x2m], weights=weights, axis=0)  # es has -inf here.
    av = jnp.average([x1, x2], weights=weights, axis=0)  # take average in normal space
    esout = jnp.exp(es)
    esout = jnp.where(~jnp.isfinite(es), av, esout)  # replace
    return esout


@cached(max_size=128)
def inpaint(ar):
    ar_floor = np.nanmin(ar)
    ar_ceil = np.nanmax(ar)
    gray = np_to_gray(ar, floor=ar_floor, ceil=ar_ceil)
    mask = (~np.isfinite(ar) * 255).astype(np.uint8)
    out = cv2.inpaint(gray, mask, 3, cv2.xphoto.INPAINT_FSR_BEST)
    out = mm_scale(out, floor=ar_floor, ceil=ar_ceil)
    return out


@cached(max_size=128)
def calc_opt_flow(gray1, gray2, mode="disflow"):
    flow = None
    if mode == "franeback":
        flow = cv2.calcOpticalFlowFarneback(
            gray1,
            gray2,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
        )
    elif mode == "denseflow":
        flow = cv2.optflow.calcOpticalFlowSparseToDense(gray1, gray2)
    elif mode == "deepflow":
        flowfun = cv2.optflow.createOptFlow_DeepFlow()
    elif mode == "pcaflow":
        # best speed and accuracy tradeoff
        flowfun = cv2.optflow.createOptFlow_PCAFlow()
    elif mode == "disflow":
        # fastest
        # https://arxiv.org/pdf/1603.03590.pdf
        flowfun = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        flowfun.setUseSpatialPropagation(True)
    if flow is None:
        flow = flowfun.calc(gray1, gray2, None)
    return flow


@jit
def _interpolate_frames(ts_weight, XY, I1, I2, flowF, flowB):
    XYW = XY + ts_weight * flowB[:, :, 0:2]
    XYW = [XYW[:, :, 1], XYW[:, :, 0]]
    I1_warped = jnp.reshape(
        jndi.map_coordinates(I1, XYW, order=1, mode="constant", cval=jnp.nan), I1.shape
    )
    XYW = XY + (1.0 - ts_weight) * flowF[:, :, 0:2]
    XYW = [XYW[:, :, 1], XYW[:, :, 0]]
    I2_warped = jnp.reshape(
        jndi.map_coordinates(I2, XYW, order=1, mode="constant", cval=jnp.nan), I2.shape
    )
    # weighted geometric mean
    return geometric_blend(I1_warped, I2_warped, weights=[1 - ts_weight, ts_weight])


@jit
def interpolate_frames(I1, I2, flowF, flowB=None, n=5, tws=None):
    """
    Interpolate frames between two images using forward and backward
    motion fields, flowF and flowB, respectively, computed from two images.
    Defaults to 5 equally spaced linear weights between frames.
    If tws, the time weights, are specified, only computes those.
    Inspired by pyoptflow.

    Parameters
    ----------
    I1: array
        First numpy array image
    I2: array
        Second numpy array image
    flowF: array
        Forward input flow motion field
    flowB: array
        Backward input flow motion field
    n: int
        Number of frames to interpolate between the images
    tws: list
        If time weights between the two slices (ie [.1, .2, etc]) are passed only those frames are computed.
    """
    if flowB == None:
        flowB = -flowF
    X, Y = jnp.meshgrid(jnp.arange(jnp.size(I1, 1)), jnp.arange(jnp.size(I1, 0)))
    XY = jnp.dstack([X, Y])
    I_result = []
    if tws is None:
        tws = jnp.arange(1, n + 1) / (n + 1)
    [
        I_result.append(
            _interpolate_frames(
                ts_weight=tw, XY=XY, I1=I1, I2=I2, flowF=flowF, flowB=flowB
            )
        )
        for tw in tws
    ]
    return I_result


@jit
def _semilagrangian(XY, delta_t, flow_tot, flow_inc, flow):
    XYW = XY + flow_tot - flow_inc / 2.0
    XYW = [XYW[:, :, 1], XYW[:, :, 0]]
    ux = delta_t * jnp.reshape(
        jndi.map_coordinates(
            flow[:, :, 0], XYW, order=1, mode="constant", cval=jnp.nan
        ),
        flow.shape[0:2],
    )
    flow_inc = jax.ops.index_update(flow_inc, jax.ops.index[:, :, 0], ux)
    vx = delta_t * jnp.reshape(
        jndi.map_coordinates(
            flow[:, :, 1], XYW, order=1, mode="constant", cval=jnp.nan
        ),
        flow.shape[0:2],
    )
    flow_inc = jax.ops.index_update(flow_inc, jax.ops.index[:, :, 1], vx)
    return flow_inc


@jit
def semilagrangian(I1, flow, t, n_steps=1, n_iter=3, inverse=True):
    """
    Apply semi-Lagrangian extrapolation to an image by using a motion field.
    Inspired by pyoptflow.

    Parameters
    ----------
    I1: array
        Input numpy image
    flow: array
        Input flow motion field
    t: float
        Time step length for extrapolation. 1.0 would be the step delta the motion flow array was computed from.
    n_steps: int
        Number of intermediate time steps to use in the extrapolation calculation.
    n_iter: int
        Number of iterations of the semilagrangian calculation.
    inverse: bool
        True means extrapolation trajectory is computed backwards along the flow. Typically gives better results.
    """
    coeff = 1.0 if inverse == False else -1.0

    delta_t = 1.0 * t / n_steps
    X, Y = jnp.meshgrid(jnp.arange(jnp.size(I1, 1)), jnp.arange(jnp.size(I1, 0)))
    XY = jnp.dstack([X, Y])

    flow_tot = jnp.zeros((flow.shape[0], flow.shape[1], 2))
    for i in range(n_steps):
        flow_inc = jnp.zeros(flow_tot.shape)
        for j in range(n_iter):
            flow_inc = _semilagrangian(
                XY=XY, delta_t=delta_t, flow_tot=flow_tot, flow_inc=flow_inc, flow=flow
            )
        flow_tot = jnp.add(flow_tot, coeff * flow_inc)

    XYW = XY + flow_tot
    XYW = [XYW[:, :, 1], XYW[:, :, 0]]
    IW = jnp.reshape(
        jndi.map_coordinates(I1, XYW, mode="constant", cval=jnp.nan, order=1), I1.shape
    )
    return IW
