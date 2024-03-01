from typing import Any, ClassVar, Dict

import cv2
import numpy as np

from base_models import BaseModelPolygon
# from shape_2d import Shape2D

__all__ = ["Polygon"]


class Polygon(BaseModelPolygon):
    """Polygon"""

    shape_type: ClassVar[str] = "polygon"

    def __init__(self, coordinates: np.ndarray):
        """Instantiates a Polygon.

        Args:
            coordinates: An N x 2 array containing the corners of the polygon
                in [x, y] format.
        """

        super(BaseModelPolygon, self).__init__(  # pylint: disable=bad-super-call
            coordinates=coordinates,
        )

    @classmethod
    def from_mask(cls, mask: np.ndarray, approx_level: float = 0.01):

        mask_as_uint = mask.astype(np.uint8) * 255

        # JANK OPEN CV VERSION CHECK to account for different cv2.findContours
        # returned tuple size
        contours_tuple = cv2.findContours(
            mask_as_uint, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours_tuple) == 2:
            contours, _ = contours_tuple
        else:
            _, contours, _ = contours_tuple

        if len(contours) > 0:
            cnt = np.concatenate(contours, axis=0)
        else:
            return Polygon(np.empty((0, 2), dtype=np.int))

        # FIXME Don't convexify by default?
        if len(contours) > 1:
            # See if I can reduce the number of points:
            cnt = cv2.convexHull(cnt)

        epsilon = approx_level * cv2.arcLength(cnt, True)
        _cnt = cv2.approxPolyDP(cnt, epsilon, True)
        return Polygon(np.squeeze(_cnt))
