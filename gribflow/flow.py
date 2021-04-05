import cv2
import jax.numpy as jnp
import jax.scipy.ndimage as jndi
import numpy as np
from jax import jit
from PIL import Image


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


def gray_to_np(ar, floor, ceil):
    ar = (ar - np.nanmin(ar)) / (np.nanmax(ar) - np.nanmin(ar))
    ar = ((ceil - floor) * ar) + floor
    return ar


def calc_opt_flow(gray1, gray2, mode="pcaflow"):
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
        # fastest but large errors
        # https://arxiv.org/pdf/1603.03590.pdf
        flowfun = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
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
    return (1.0 - ts_weight) * I1_warped + ts_weight * I2_warped


def interpolate_frames(I1, I2, flowF, flowB=None, n=5, tws=None):
    """
    Interpolate frames between two images using forward and backward 
    motion fields, flowF and flowB, respectively, computed from two images.
    Defaults to 5 equally spaced linear weights between frames.
    If tws, the time weights, are specified, only computes those.
    Inspired by pyoptflow.
    """
    if flowB == None:
        flowB = -flowF
    X, Y = jnp.meshgrid(np.arange(np.size(I1, 1)), jnp.arange(jnp.size(I1, 0)))
    XY = jnp.dstack([X, Y])
    I_result = []
    if tws is None:
        tws = np.arange(1, n + 1) / (n + 1)
    for tw in tws:
        I_result.append(
            _interpolate_frames(
                ts_weight=tw, XY=XY, I1=I1, I2=I2, flowF=flowF, flowB=flowB
            )
        )
    return I_result


def semilagrangian(I, flow, t, n_steps, n_iter=3, inverse=True):
    """
    Apply semi-Lagrangian extrapolation to an image by using a motion field.
    Inspired by pyoptflow.
    """
    coeff = 1.0 if inverse == False else -1.0

    delta_t = 1.0 * t / n_steps
    X, Y = jnp.meshgrid(jnp.arange(jnp.size(I, 1)), jnp.arange(jnp.size(I, 0)))
    XY = jnp.dstack([X, Y])

    flow_tot = np.zeros((flow.shape[0], flow.shape[1], 2))
    for i in range(n_steps):
        flow_inc = np.zeros(flow_tot.shape)
        for j in range(n_iter):
            XYW = XY + flow_tot - flow_inc / 2.0
            XYW = [XYW[:, :, 1], XYW[:, :, 0]]
            flow_inc[:, :, 0] = delta_t * jnp.reshape(
                jndi.map_coordinates(
                    flow[:, :, 0], XYW, order=1, mode="constant", cval=jnp.nan
                ),
                flow.shape[0:2],
            )
            flow_inc[:, :, 1] = delta_t * jnp.reshape(
                jndi.map_coordinates(
                    flow[:, :, 1], XYW, order=1, mode="constant", cval=jnp.nan
                ),
                flow.shape[0:2],
            )

        flow_tot += coeff * flow_inc

    XYW = XY + flow_tot
    XYW = [XYW[:, :, 1], XYW[:, :, 0]]
    IW = jnp.reshape(
        jndi.map_coordinates(I, XYW, mode="constant", cval=jnp.nan, order=1), I.shape
    )
    return IW
