import cv2 as cv  # noqa
import numpy as np
from numpy.typing import NDArray
from typing import Callable

# load reference image to optimize against
# reference_image = Image.open("src/img/example_001.png").convert("RGB")


class Canvas:
    def __init__(self, resolution=(600, 800), bg_color=(255, 255, 255)) -> None:
        self.resolution = resolution
        self.bg_color = bg_color
        self._canvas: NDArray | None = None
        # setup blank canvas
        self.clear()
        self._cache: NDArray = self._canvas
        # instatiate Drawer class
        self.draw = Drawer(self)

    @property
    def canvas(self):
        return self._canvas

    def clear(self):
        """Create a blank canvas with the specified background color"""
        self._canvas = np.ones(self.resolution + (3,), np.uint8) * np.array(self.bg_color, np.uint8)

    def revert(self):
        """revert canvas to the cache state"""
        self._canvas = self._cache.copy()


class Drawer:
    def __init__(self, parent: "Canvas") -> None:
        self.parent = parent

    def _cache_canvas_state(method: Callable) -> Callable:
        """decorator to cache the state of the canvas before drawing"""

        def wrapper(self, *args, **kwargs):
            # save curreng state of canvas to _cache before drawing
            self.parent._cache = self.parent._canvas.copy()
            return method(self, *args, **kwargs)

        return wrapper

    @_cache_canvas_state
    def line(self, pt1, pt2, color=(255, 255, 255), thickness=-1):
        # Draw a line on the parent canvas
        cv.line(self.parent._canvas, pt1, pt2, color, thickness)

    @_cache_canvas_state
    def rectangle(self, pt1, pt2, color=(0, 0, 0), thickness=-1):
        # Draw a rectangle on the parent canvas
        cv.rectangle(self.parent._canvas, pt1, pt2, color, thickness)

    @_cache_canvas_state
    def circle(self, center, radius, color=(0, 0, 0), thickness=-1):
        # Draw a circle on the parent canvas
        cv.circle(
            self.parent._canvas,
            center,
            radius,
            color,
            thickness,
        )
