from typing import List, NamedTuple, Tuple

import numpy as np


class RLEMask(NamedTuple):
    """Run-length-encoding for masks.

    Run-length-encoding is a simple compression method for masks.
    The cocoapi documentation explains it well:
    - https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py
    This implementation differs from the COCO implementation by using
    row-major traversal, rather than column-major as in COCO.
    Row-major representation is more efficient for frontend
    applications passing arrays to/from html canvas.

    NOTE: column-major encoding can be achieved by simply transposing the mask
      before passing to `encode`. Take care that any data produced has its
      encoding details clearly documented.
    """

    run_lengths: List[int]
    shape: Tuple[int, int]

    @classmethod
    def encode(cls, mask: np.ndarray) -> "RLEMask":
        """Row-major run-length-encoding of 2D binary mask.

        For example, given the flattened mask [0 0 1 1 1 0 1], the run lengths
        would be [2 3 1 1], or for the flattened mask [1 1 1 1 1 1 0] the
        run lengths would be [0 6 1].
        If the mask starts with '1', the encoding starts with a zero.

        Args:
          mask: 2D binary mask to be encoded

        Returns:
          RLEMask object for the encoded mask
        """
        if len(mask.shape) != 2:
            raise ValueError("RLE is only defined for 2D masks")
        if np.issubdtype(mask.dtype, np.integer):
            if len(np.unique(mask)) > 2:
                raise ValueError("RLE is only defined for binary masks")
        elif mask.dtype != "bool":
            raise ValueError("RLE is only defined for integer and boolean masks")

        flattened = mask.ravel(order="C")  # Row-major
        # Diff the array so that only the change points are non-zero,
        # then find the indices of the change points
        (change_indices,) = np.nonzero(flattened[:-1] != flattened[1:])
        change_indices = np.concatenate(([-1], change_indices, [len(flattened) - 1]))
        # Diff the array of change-point indices to find the run-lengths
        run_lengths = change_indices[1:] - change_indices[:-1]
        # If the original array starts with a 1 or True, begin the run lengths
        # with a zero
        if flattened[0]:
            run_lengths = np.concatenate(([0], run_lengths))
        return cls(run_lengths=run_lengths.tolist(), shape=list(mask.shape))

    def decode(self) -> np.ndarray:
        """Decode run-length-encoding back into 2D binary mask

        Returns:
          2D binary mask array
        """
        mask_sections = []
        mask_value = 0
        for run_length in self.run_lengths:
            mask_sections.append([mask_value] * run_length)
            mask_value = 1 if mask_value == 0 else 0
        return np.concatenate(mask_sections).reshape(self.shape)
