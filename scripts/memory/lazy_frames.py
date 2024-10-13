import numpy as np
from typing import List, Optional, Union


class LazyFrames:
    def __init__(self, frames: List[np.ndarray]):
        """
        Initialize LazyFrames with a list of frames.

        Args:
            frames (List[np.ndarray]): A list of frames to stack.
        """
        self._frames: Optional[List[np.ndarray]] = frames
        self._out: Optional[np.ndarray] = None

    def _force(self) -> np.ndarray:
        """
        Force the stacking of frames if not already done.

        Returns:
            np.ndarray: The stacked frames.
        """
        if self._out is None:
            # Concatenate frames along the first axis (channels)
            self._out = np.stack(self._frames, axis=0)
            self._frames = None  # Free up memory
        return self._out

    def __array__(self, dtype: Optional[Union[np.dtype, str]] = None) -> np.ndarray:
        """
        Convert the LazyFrames to a numpy array.

        Args:
            dtype (Optional[Union[np.dtype, str]]): The desired data type of the array.

        Returns:
            np.ndarray: The stacked frames as a numpy array.
        """
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self) -> int:
        """
        Get the number of frames.

        Returns:
            int: The number of frames.
        """
        return len(self._force())

    def __getitem__(self, idx: int) -> np.ndarray:
        """
        Get a specific frame by index.

        Args:
            idx (int): The index of the frame to retrieve.

        Returns:
            np.ndarray: The frame at the specified index.
        """
        return self._force()[idx]
