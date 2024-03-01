# pylint: disable=unnecessary-pass
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import numpy as np



__all__ = ["Shape2D"]


class Shape2D(ABC):
    @abstractmethod
    def draw_onto(
        self,
        image: np.ndarray,
        colour: Tuple[int, int, int] = (0, 255, 0),
        alpha: float = 0.5,
        **kwargs: Any,
    ) -> np.ndarray:
        """Draws the shape onto an image.

        Args:
            image (np.array): A numpy array containing the image.
            kwargs (Any): Other arguments for drawing, which may be
                shape-specific.

        Returns:
            np.array: The image with the annotation drawn on it.
        """
        pass

    @property
    @abstractmethod
    def bounding_rectangle(self) -> np.ndarray:
        """Returns the coordinates of the shape's bounding rectangle. The
        result will be a numpy array of shape [4,] in [x1, y1, x2, y2] form."""
        pass

    @abstractmethod
    def shift_by(self, x_shift: int, y_shift: int) -> None:
        """Shifts the annotation by x_shift in the x direction and y_shift in
        the y direction."""
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Produces a dictionary version of the shape which can be JSON
        encoded."""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, shape_dict: Dict[str, Any]):
        """Constructs the shape from its dictionary-encoding."""
        pass


