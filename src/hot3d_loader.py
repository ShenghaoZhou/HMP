import hand_tracking_toolkit
from hand_tracking_toolkit.dataset import (
    HandShapeCollection, decode_hand_pose, decode_cam_params, HandSide
)
# from hand_tracking_toolkit.hand_models.umetrack_hand_model import (
#     from_json as from_umetrack_hand_model_json,
# )

from hand_tracking_toolkit.hand_models.mano_hand_model import (
    forward_kinematics as mano_forward_kinematics,
    MANOHandModel
)
import hand_tracking_toolkit.math_utils
import torch
import json
from fitting_utils import get_joints2d
from rotations import axis_angle_to_matrix, matrix_to_rotation_6d
from target_data import HMPData
from pathlib import Path
from body_model.mano import BodyModel
from nemf.fk import ForwardKinematicsLayer

MANO_DIR = Path("./data/body_models/mano/")
MANO_RH_DIR = MANO_DIR / "MANO_RIGHT.pkl"
RIGHT_HAND_INDEX = 1
stream_id = "214-1"
large_f = torch.tensor([5000., 5000.])


class Object(object):
    pass


args = Object()
setattr(args, "smpl", Object())
setattr(args.smpl, "smpl_body_model", str(MANO_DIR))
fk = ForwardKinematicsLayer(args)


def load_json(file_name):
    with open(file_name) as f:
        return json.load(f)


def load_hand_shape(filename="__hand_shapes.json__"):
    shape_params_dict = load_json(filename)
    return HandShapeCollection(
        mano_beta=torch.tensor(shape_params_dict["mano"]),
        umetrack=None
        # from_umetrack_hand_model_json(
        #     shape_params_dict["umetrack"]),
    )


def load_cameras(camera_json_file):
    cameras_raw = load_json(camera_json_file)
    cameras = {}
    Ts_device_from_camera = {}
    for stream_key, camera_raw in cameras_raw.items():
        cameras[stream_key] = hand_tracking_toolkit.camera.from_json(
            camera_raw)
        pose = camera_raw["calibration"]["T_device_from_camera"]
        Ts_device_from_camera[stream_key] = hand_tracking_toolkit.math_utils.quat_trans_to_matrix(
            *pose["quaternion_wxyz"], *pose["translation_xyz"]
        )
    return cameras, Ts_device_from_camera


def get_number_of_frames(path):
    return max([int(name.name.split(".info.json")[0]) for name in path.glob("*.info.json")]) + 1


def load_hot3d_folder(folder_path):
    folder = Path(folder_path)
    hand_shape = load_hand_shape(folder / "__hand_shapes.json__")

    # detailed hand pose joint data
    pos_all = []
    global_xform_all = []
    # hand wrist pose
    trans_all = []
    root_oritent_all = []
    # camera parameters
    cam_Rs = []
    cam_ts = []
    cam_fs = []
    # cam_centers = []
    cam_center = None
    # hand joint
    joints2d = []
    joints3d = []
    # hand shape
    betas = []

    num_frames = get_number_of_frames(folder)
    mano_hand_model_toolbox = MANOHandModel(str(MANO_DIR))

    # notice the subtle difference: hmp MANO model sets flat_hand_mean, which assumes hand_mean is 0
    mano_hand_model_hmp = BodyModel(model_type="mano", model_path=str(MANO_RH_DIR), device='cpu',
                                    **{"flat_hand_mean": True, "use_pca": False, "is_rhand": True})
    if num_frames > 128:
        # FIXME: later, we should extend the code for any num_frames, by breaking it to 128 chunks
        # see the logic in fitting_app.py: 829
        print("warning: the method operates on a batch of 128 frames, so only the first ones are used")
        num_frames = 128
    for frame_id in range(num_frames):
        # hand shape
        with open(folder / f"{frame_id:06d}.cameras.json") as f:
            j = json.load(f)
            cameras = decode_cam_params(j)
        camera = cameras[stream_id]
        cam_fs.append(torch.tensor(camera.f))
        # cam_centers.append(torch.tensor(camera.c))
        cam_center = torch.tensor(camera.c)
        # camera pose
        R_cam2world = torch.from_numpy(camera.T_world_from_eye[:3, :3])
        t_cam_in_world = torch.from_numpy(camera.T_world_from_eye[:3, 3])
        # camera extrinsics
        R_world2cam = R_cam2world.T
        t_world_in_cam = -R_world2cam @ t_cam_in_world
        # cam_R.append(R_cam2world)
        # cam_t.append(t_cam_in_world)
        cam_Rs.append(R_cam2world)
        cam_ts.append(t_world_in_cam)

        betas.append(hand_shape.mano_beta)
        # j = json.loads(folder / f"{frame_id:06d}.hands.json")
        # hand_pose = decode_hand_pose(j)
        with open(folder / f"{frame_id:06d}.hands.json") as f:
            hand_pose = decode_hand_pose(json.load(f))
        # handle two hands for each frame
        for hand_side in hand_pose.keys():
            if hand_side == HandSide.LEFT:
                # for now, we only consider right hand
                continue

            # we will stick with MANO convention for 21 joints, so we avoid mano_forward_kinematics
            # for its extra mapping
            # right_hand_mask = torch.tensor(
            #     hand_side == HandSide.RIGHT, dtype=torch.bool)
            wrist_xform = hand_pose[hand_side].mano.wrist_xform
            mano_theta = hand_pose[hand_side].mano.mano_theta
            # _, landmarks = mano_hand_model_toolbox(
            #     mano_beta=hand_shape.mano_beta,
            #     mano_theta=mano_theta,
            #     wrist_xform=wrist_xform,
            #     is_right_hand=right_hand_mask,
            # )
            # in non-batch mode, landmarks just repeat in the first dim
            # landmarks = landmarks[0]  # (21, 3)
            # mano_theta = hand_pose[hand_side].mano.mano_theta
            # res = mano_forward_kinematics(
            #     hand_pose=hand_pose[hand_side].mano,
            #     mano_beta=hand_shape.mano_beta,
            #     mano_model=mano_hand_model_toolbox
            # )
            # print(res)
            # batch version of mano fk from hand_tracking_toolkit;
            # notice, it is fundamentally the same as mano fk for HMP, which use the same underlying smplx model code
            # verts, landmarks = mano_hand_model(
            #     mano_beta,
            #     mano_theta,
            #     wrist_xform,
            #     is_right_hand=right_hand_mask,
            # )
            # mano_forward_kinematics(
            #     hand_pose=mano_theta,
            #     mano_beta=hand_shape.mano_beta,
            #     mano_model=mano_hand_model
            # )

            # follow implementation in
            # hand_tracking_toolkit.hand_models.mano_hand_model.MANOHandModel::forward()
            global_orient = wrist_xform[:3]
            transl = wrist_xform[3:]
            # in toolbox, pose is defined in pca components as mano_theta
            # we need to upgrade it to full 45-dim pose
            hand_model_layer_toolbox = mano_hand_model_toolbox.mano_layer_right if hand_side == HandSide.RIGHT \
                else mano_hand_model_toolbox.mano_layer_left
            hand_pose = mano_theta @ hand_model_layer_toolbox.hand_components
            # hand pose from toolbox has added hand_mean, to work with hmp mano, we need to remove it
            hand_pose = hand_pose + hand_model_layer_toolbox.hand_mean
            # feed mano theta to fk, to obtain joint pose
            # checked: run mano_hand_model_hmp's fk will give the same joint3d as the hand_model_toolbox

            mano_out = mano_hand_model_hmp(
                input_dict={
                    "betas": hand_shape.mano_beta.reshape(1, -1),  # (1, 10)
                    "hand_pose": hand_pose.reshape(1, -1),  # (1, 45)
                    "global_orient": global_orient.reshape(1, -1),  # (1, 3)
                    "no_shift": True,
                    "return_finger_tips": True,
                    "transl":  transl.reshape(1, -1)  # (1, 3)
                }
            )
            # TODO: double check if this joint3d makes sense. We can get expected joint3d from toolbox(which go through diffent handmodel fk)
            joint3d = mano_out.joints.view(1, -1, 21, 3)
            # TODO: this step is nothing more than standard camera model projection. Since we know the camera model, it should be easy
            # we shouldn't use the function provided in HMP, since it assumes plain camera model.

            # FIXME: the projected 2d coodrinate is too large to make sense. Something is wrong.

            # inside get_joint2d, it seems that cam_R.T is used as rotation
            joint2d = get_joints2d(
                joints3d_pred=joint3d,
                cam_t=cam_ts[-1],
                cam_R=cam_Rs[-1],
                cam_f=cam_fs[-1],
                cam_center=cam_center
            )
            # (x, y, conf)
            joint2d_with_conf = torch.cat(
                [joint2d, torch.ones_like(joint2d[..., 0:1])], dim=-1)
            # this is projection with exact fisheye camera model
            # joint2d = camera.world_to_window(joint3d.numpy())
            # to see how it is constructed, refer to datasets/amass.py::get_stage2_res
            # 16 = 1 wrist + 15 finger tips
            # root_orient is prepended to the joints
            full_pose = torch.cat([global_orient, hand_pose]).reshape(
                1, 16, 3)  # (1, 48)
            rotmat = axis_angle_to_matrix(full_pose)  # (B, J=16, 3, 3)
            rot6d = matrix_to_rotation_6d(rotmat)
            pos, global_xform = fk(rot6d.to(fk.device))
            global_xform_6d = matrix_to_rotation_6d(global_xform[..., :3, :3])

            pos_all.append(pos)
            global_xform_all.append(global_xform_6d)
            trans_all.append(transl)
            global_orient_6d = matrix_to_rotation_6d(
                axis_angle_to_matrix(global_orient))
            root_oritent_all.append(global_orient_6d)

            joints3d.append(joint3d)
            joints2d.append(joint2d_with_conf)

    # shape is expected to be (N, 128, ...), i.e. split to batch of size 128
    data = HMPData(
        pos=torch.stack(pos_all, dim=1),
        global_xforms=torch.stack(global_xform_all, dim=1),
        trans=torch.stack(trans_all).unsqueeze(0),
        root_orient=torch.stack(root_oritent_all).unsqueeze(0),
        cam_R=torch.stack(cam_Rs).unsqueeze(0),
        cam_t=torch.stack(cam_ts).unsqueeze(0),
        cam_f=torch.stack(cam_fs).unsqueeze(0),
        cam_center=cam_center.unsqueeze(0),
        img_dir=str(folder),
        joint2d=torch.cat(joints2d, dim=1),
        joint3d=torch.cat(joints3d, dim=1),
        betas=torch.stack(betas).unsqueeze(0),
    )
    return data


if __name__ == "__main__":
    load_hot3d_folder("../hot3d_clips_data/clip-001849")
