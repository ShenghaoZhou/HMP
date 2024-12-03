import dataclasses
import torch


@dataclasses.dataclass
class HMPData:
    # hand joint pose, including joint pos, and
    pos: torch.Tensor  # joint pos from fk
    # global rotmat for joint (J, 6), in rotation6d format
    global_xforms: torch.Tensor
    # wrist pose
    trans: torch.Tensor
    root_orient: torch.Tensor  # in matrix
    # camera pose
    cam_R: torch.Tensor
    cam_t: torch.Tensor
    # camera intrinsics
    cam_f: torch.Tensor
    cam_center: torch.Tensor
    # # image info
    # img_width: int
    # img_height: int
    img_dir: str
    # frame_id: int
    # save_path: str
    # hand prediction
    joint2d: torch.Tensor
    joint3d: torch.Tensor
    betas: torch.Tensor
    # handedness: str  # left or right
    # mediapipe_bbox_conf: float
    # # meta info
    # config_type: str
