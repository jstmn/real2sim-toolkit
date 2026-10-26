from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsics."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self) -> None:
        assert isinstance(self.width, int) and self.width > 0, f"width must be positive int, got {self.width}"
        assert isinstance(self.height, int) and self.height > 0, f"height must be positive int, got {self.height}"
        assert self.fx > 0 and self.fy > 0, f"fx/fy must be positive, got {self.fx}/{self.fy}"

    @property
    def values(self) -> tuple[float, float, float, float]:
        return (self.fx, self.fy, self.cx, self.cy)

    @property
    def intrinsic_matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float64)

    @classmethod
    def from_matrix(cls, matrix: np.ndarray, width: int, height: int) -> "CameraIntrinsics":
        assert matrix.shape == (3, 3), f"matrix must be 3x3, got {matrix.shape}"
        return cls(
            width=width,
            height=height,
            fx=float(matrix[0, 0]),
            fy=float(matrix[1, 1]),
            cx=float(matrix[0, 2]),
            cy=float(matrix[1, 2]),
        )


@dataclass
class CameraExtrinsics:
    """Camera extrinsic: 4x4 world_T_cam."""

    matrix: np.ndarray
    parent_frame: str = "world"
    child_frame: str = "camera"

    def __post_init__(self) -> None:
        assert isinstance(self.matrix, np.ndarray), f"matrix must be ndarray, got {type(self.matrix)}"
        assert self.matrix.shape == (4, 4), f"matrix must be 4x4, got {self.matrix.shape}"
        assert self.matrix.dtype in (
            np.float32,
            np.float64,
        ), f"matrix dtype must be float32/64, got {self.matrix.dtype}"
        if not self.matrix.flags["C_CONTIGUOUS"]:
            self.matrix = np.ascontiguousarray(self.matrix)


@dataclass(frozen=True)
class VerticalCylinder:
    """Vertical cylinder centered at `(x_m, y_m)` spanning `[min_z_m, max_z_m]`."""

    x_m: float
    y_m: float
    radius_m: float
    min_z_m: float
    max_z_m: float

    def __post_init__(self) -> None:
        assert self.radius_m > 0.0, f"radius_m must be > 0, got {self.radius_m}"
        assert self.min_z_m < self.max_z_m, f"min_z_m {self.min_z_m} must be < max_z_m {self.max_z_m}"

    def contains(self, points: np.ndarray) -> np.ndarray:
        """`(N,)` bool: which of the `(N, 3)` points lie inside the cylinder (inclusive)."""
        assert points.ndim == 2 and points.shape[1] == 3, f"points must be (N, 3), got {points.shape}"
        r_sq = (points[:, 0] - self.x_m) ** 2 + (points[:, 1] - self.y_m) ** 2
        return (r_sq <= self.radius_m**2) & (points[:, 2] >= self.min_z_m) & (points[:, 2] <= self.max_z_m)


@dataclass
class WorkspaceBounds:
    """Axis-aligned box minus `excluded` regions (e.g. camera mounts standing inside the box)."""

    min_x_m: float
    max_x_m: float
    min_y_m: float
    max_y_m: float
    min_z_m: float
    max_z_m: float
    excluded: tuple[VerticalCylinder, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        assert self.min_x_m < self.max_x_m
        assert self.min_y_m < self.max_y_m
        assert self.min_z_m < self.max_z_m
        assert all(
            isinstance(c, VerticalCylinder) for c in self.excluded
        ), f"excluded must hold VerticalCylinders, got {self.excluded}"

    @property
    def extents(self) -> tuple[float, float, float]:
        return (self.max_x_m - self.min_x_m, self.max_y_m - self.min_y_m, self.max_z_m - self.min_z_m)

    @property
    def center(self) -> tuple[float, float, float]:
        return (
            (self.max_x_m + self.min_x_m) / 2,
            (self.max_y_m + self.min_y_m) / 2,
            (self.max_z_m + self.min_z_m) / 2,
        )

    def padded(self, margin_m: float) -> "WorkspaceBounds":
        """These bounds grown by `margin_m` on every side. `excluded` regions are kept as is."""
        assert margin_m >= 0.0, f"margin_m must be >= 0, got {margin_m}"
        return WorkspaceBounds(
            min_x_m=self.min_x_m - margin_m,
            max_x_m=self.max_x_m + margin_m,
            min_y_m=self.min_y_m - margin_m,
            max_y_m=self.max_y_m + margin_m,
            min_z_m=self.min_z_m - margin_m,
            max_z_m=self.max_z_m + margin_m,
            excluded=self.excluded,
        )

    def contains(self, points: np.ndarray) -> np.ndarray:
        """`(N,)` bool: which of the `(N, 3)` points lie inside the box (inclusive) and outside every `excluded`
        region."""
        assert points.ndim == 2 and points.shape[1] == 3, f"points must be (N, 3), got {points.shape}"
        lo = np.array([self.min_x_m, self.min_y_m, self.min_z_m])
        hi = np.array([self.max_x_m, self.max_y_m, self.max_z_m])
        inside = np.all((points >= lo) & (points <= hi), axis=1)
        for region in self.excluded:
            inside &= ~region.contains(points)
        return inside


@dataclass
class CameraImage:
    """A single camera's observation of a scene, with a segmentation mask for one object."""

    camera_name: str
    image: np.ndarray
    mask: np.ndarray

    def __post_init__(self) -> None:
        assert isinstance(self.camera_name, str) and len(self.camera_name) > 0
        assert isinstance(self.image, np.ndarray) and self.image.ndim == 3
        assert isinstance(self.mask, np.ndarray) and self.mask.ndim == 2
        assert (
            self.mask.shape == self.image.shape[:2]
        ), f"mask shape {self.mask.shape} != image shape {self.image.shape[:2]}"


@dataclass
class ObjectAssets:
    """Assets and observations for a single detected object, across one or more camera views."""

    object_name: str
    camera_images: list[CameraImage]
    glb_filepath: Path | None = None
    obj_filepath: Path | None = None
    fbx_filepath: Path | None = None
    usdz_filepath: Path | None = None
    pointcloud: np.ndarray | None = None

    def __post_init__(self) -> None:
        assert isinstance(self.object_name, str) and len(self.object_name) > 0
        assert (
            isinstance(self.camera_images, list) and len(self.camera_images) > 0
        ), "camera_images must be a non-empty list"
        assert all(
            isinstance(camera_image, CameraImage) for camera_image in self.camera_images
        ), f"All camera_images entries must be CameraImage, got {[type(c) for c in self.camera_images]}"


@dataclass
class ObjectPose:
    """Pose estimate for a single object."""

    object_name: str
    pose_cam: np.ndarray
    pose_world: np.ndarray
    mesh_path: Path
    mask: np.ndarray

    def __post_init__(self) -> None:
        assert self.pose_cam.shape == (4, 4), f"pose_cam must be 4x4, got {self.pose_cam.shape}"
        assert self.pose_world.shape == (4, 4), f"pose_world must be 4x4, got {self.pose_world.shape}"
