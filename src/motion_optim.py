"""
This is a revised version of fitting_app.py:motion_reconstruction() function,
with a clear input and output.

Conceptually, it takes a joint camera and hand pose trajctory, 
and it returns a refined hand pose trajectory, via the latent optimization of learned NeMF.

"""
import os
import time
from dataclasses import dataclass

from loguru import logger
import joblib
import torch
import torch.nn.functional as F
import torch.nn as nn

from fitting_utils import get_joints2d, joints2d_loss
from hot3d_loader import load_hot3d_folder
from nemf.generative import Architecture
from nemf.fk import ForwardKinematicsLayer
from rotations import rotation_6d_to_matrix, matrix_to_axis_angle, matrix_to_rotation_6d
from arguments import Arguments
from nemf.losses import GeodesicLoss, pos_smooth_loss, rot_smooth_loss
from utils import align_joints, build_canonical_frame, estimate_angular_velocity, estimate_linear_velocity, normalize
from target_data import HMPData
from body_model.mano import BodyModel


HAND_JOINT_NUM = 16


def forward_mano(output, fk, hand_model):
    rotmat = output['rotmat']  # (B, T, J, 3, 3)
    B, T, J, _, _ = rotmat.size()

    # b_size, _, n_joints = rotmat.shape[:3]
    local_rotmat = fk.global_to_local(
        rotmat.view(-1, J, 3, 3))  # (B x T, J, 3, 3)
    local_rotmat = local_rotmat.view(B, -1, J, 3, 3)

    root_orient = rotation_6d_to_matrix(output['root_orient'])  # (B, T, 3, 3)

    local_rotmat[:, :, 0] = root_orient

    poses = matrix_to_axis_angle(local_rotmat)  # (T, J, 3)
    poses = poses.view(-1, J * 3)

    # no_shift is the flag for not shifting the wrist location to the origin
    mano_out = hand_model(input_dict={"betas": output['betas'].view(-1, 10),
                                      "global_orient": poses[..., :3].view(-1, 3),
                                      "hand_pose": poses[..., 3:].view(-1, 45),
                                      "no_shift": True,
                                      "return_finger_tips": True,
                                      "transl": output['trans'].view(-1, 3)})

    return mano_out


def L_rot(pred, gt, T, conf=None, use_geodesic_loss=True, use_l1_loss=True):
    """
    Args:
        source, target: rotation matrices in the shape B x T x J x 3 x 3.
        T: temporal masks.

    Returns:
        reconstruction loss evaluated on the rotation matrices.
    """
    if conf is not None:
        criterion_rec = nn.L1Loss(
            reduction='none') if use_l1_loss else nn.MSELoss(reduction='none')
    else:
        criterion_rec = nn.L1Loss() if use_l1_loss else nn.MSELoss()
    criterion_geo = GeodesicLoss()

    B, seqlen, J, _, _ = pred.shape

    if use_geodesic_loss:
        if conf is not None:
            loss = (conf.squeeze(-1) ** 2) * criterion_geo(
                pred[:, T].view(-1, 3, 3), gt[:, T].view(-1, 3, 3), reduction='none').reshape(B, seqlen, J)
            loss = loss.mean()
        else:
            loss = criterion_geo(
                pred[:, T].view(-1, 3, 3), gt[:, T].view(-1, 3, 3))
    else:
        if conf is not None:
            loss = (conf.unsqueeze(-1) ** 2) * \
                criterion_rec(pred[:, T], gt[:, T])
            loss = loss.mean()
        else:
            loss = criterion_rec(pred[:, T], gt[:, T])

    return loss


def L_pos(source, target, T):
    """
    Args:
        source, target: joint local positions in the shape B x T x J x 3.
        T: temporal masks.

    Returns:
        reconstruction loss evaluated on the joint local positions.
    """
    criterion_rec = nn.L1Loss() if args.l1_loss else nn.MSELoss()
    loss = criterion_rec(source[:, T], target[:, T])

    return loss


def L_orient(source, target, T, bbox_conf=None, use_geodesic_loss=True):
    """
    Args:
        source: predicted root orientation in the shape B x T x 6.
        target: root orientation in the shape of B x T x 6.
        T: temporal masks.

    Returns:
        reconstruction loss evaluated on the root orientation.
    """
    criterion_rec = nn.L1Loss() if args.l1_loss else nn.MSELoss()
    criterion_geo = GeodesicLoss()

    source = rotation_6d_to_matrix(source)  # (B, T, 3, 3)
    target = rotation_6d_to_matrix(target)  # (B, T, 3, 3)

    if use_geodesic_loss:

        if bbox_conf is not None:

            loss = criterion_geo(
                source[:, T].view(-1, 3, 3), target[:, T].view(-1, 3, 3), reduction='none')
            bbox_conf_coef = bbox_conf.reshape(-1)
            loss = ((bbox_conf_coef ** 2) * loss).mean()

        else:
            loss = criterion_geo(
                source[:, T].view(-1, 3, 3), target[:, T].view(-1, 3, 3))

    else:
        loss = criterion_rec(source[:, T], target[:, T])

    return loss


def L_trans(source, target, T, bbox_conf=None, use_l1_loss=True):
    """
    Args:
        source: predict global translation in the shape B x T x 3 (the origin is (0, 0, height)).
        target: global translation of the root joint in the shape B x T x 3.
        T: temporal masks.

    Returns:
        reconstruction loss evaluated on the global translation.
    """

    trans = source
    trans_gt = target

    # dont make reduction and weight by bbox_conf
    if bbox_conf is not None:
        criterion_pred = nn.L1Loss(
            reduction='none') if use_l1_loss else nn.MSELoss(reduction='none')

        # reshape to (T * N, 3)
        loss = criterion_pred(
            trans[:, T].reshape(-1, 3), trans_gt[:, T].reshape(-1, 3)).mean(1)
        bbox_conf_coef = bbox_conf.reshape(-1)
        loss = ((bbox_conf_coef ** 2) * loss).mean()

    else:
        criterion_pred = nn.L1Loss() if use_l1_loss else nn.MSELoss()
        loss = criterion_pred(trans[:, T], trans_gt[:, T])

    return loss


def motion_prior_loss(latent_motion_pred):
    # assume standard normal
    loss = latent_motion_pred**2
    loss = torch.mean(loss)

    return loss


class HMPModel:
    def __init__(self, f=5000.0):
        # load model
        args = Arguments(
            './configs', filename='in_the_wild_sample_config.yaml')
        args.dataname = "in_the_wild"
        ngpu = 1
        model = Architecture(args, ngpu)
        model.load(optimal=True)
        model.eval()

        self.args = args
        self.args.pkl_output_dir = "/workspace/project/egocentric/HMP/my_res"
        self.model = model
        self.fk = ForwardKinematicsLayer(args)
        MANO_RH_DIR = "./data/body_models/mano/MANO_RIGHT.pkl"
        self.hand_model = BodyModel(model_type="mano", model_path=MANO_RH_DIR, device='cuda',
                                    **{"flat_hand_mean": True, "use_pca": False,
                                       #    "batch_size": args.N_frames,
                                       "is_rhand": True})
        # known camera focal length
        self.f = f

    def optim_step(self, stg_conf, stg_id, z_l, z_g, betas, target, B,
                   seqlen, trans, root_orient, init_z_l, mean_betas, T, full_cam_R, full_cam_t,
                   pose):
        logger.info(f'Running optimization stage {stg_id+1} ...')
        args = self.args
        opt_params = []
        for param in stg_conf.opt_params:
            if param == 'root_orient':
                opt_params.append(root_orient)
            elif param == 'trans':
                opt_params.append(trans)
            elif param == 'z_l':
                opt_params.append(z_l)
            elif param == 'pose':
                opt_params.append(pose)
            elif param == 'betas':
                if betas is None:
                    logger.error(
                        'Cannot optimize betas if args.opt_betas is False')
                opt_params.append(betas)
            else:
                raise ValueError(f'Unknown parameter {param}')

        for param in opt_params:
            param.requires_grad = True

        optimizer = torch.optim.Adam(opt_params, lr=stg_conf.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, args.scheduler.step_size, args.scheduler.gamma, verbose=False)

        loss_dict_by_step = {"rot": [], "reproj": [], "rot_sm": [], "orient_sm": [], 'betas_prior': [], "j3d_sm": [], "pose_prior": [],
                             "trans_sm": [], "mot_prior": [], "init_z_prior": [], "orient": [], "trans": [], "loss": []}

        # optimize the z_l and root_orient, pos, trans, 2d kp objectives
        start_time = time.time()
        for i in range(stg_conf.niters):

            optimizer.zero_grad()

            output = self.model.decode(
                z_l, z_g, length=args.data.clip_length, step=1)

            for k, v in output.items():
                if torch.isnan(v).any():
                    logger.warning(
                        f'{k} in output is NaN, skipping this stage...')
                    return 0

            output['betas'] = betas[:, None, None, :].repeat(1, B, seqlen, 1)

            # For batch optimization
            global_trans = trans.clone().reshape(args.nsubject, B, seqlen, 3)
            global_trans = global_trans.reshape(B*args.nsubject, seqlen, 3)

            output['trans'] = global_trans
            output['root_orient'] = root_orient

            rh_mano_out = forward_mano(output, self.fk, self.hand_model)
            joints3d = rh_mano_out.joints.view(B, seqlen, -1, 3)
            # vertices3d = rh_mano_out.vertices.view(B, seqlen, -1, 3)

            optim_cam_R = rotation_6d_to_matrix(full_cam_R)
            optim_cam_t = full_cam_t

            joints2d_pred = get_joints2d(joints3d_pred=joints3d,
                                         cam_t=optim_cam_t.unsqueeze(
                                             0).repeat_interleave(args.nsubject, 0),
                                         cam_R=optim_cam_R.unsqueeze(
                                             0).repeat_interleave(args.nsubject, 0),
                                         cam_f=torch.tensor([self.f, self.f]),
                                         cam_center=target['cam_center'])

            output['joints2d'] = joints2d_pred
            output['joints3d'] = joints3d

            _bbox_conf_ = None
            # _bbox_conf_[_bbox_conf_<0.6] = 0.0

            local_rotmat = self.fk.global_to_local(
                output['rotmat'].view(-1, HAND_JOINT_NUM, 3, 3))  # (B x T, J, 3, 3)
            local_rotmat = local_rotmat.view(
                B*args.nsubject, -1, HAND_JOINT_NUM, 3, 3)  # (B x T, J, 3, 3)

            local_rotmat_gt = target['rotmat']
            loss_dict = {}

            if stg_conf.lambda_rot > 0:
                mano_joint_conf = torch.zeros_like(
                    target['rotmat'][..., :1, 0])

                for si in range(16):
                    op_conf = [target['joints2d'][:, :, si, 2]]
                    max_conf = torch.stack(op_conf, dim=0).max(0).values
                    mano_joint_conf[:, :, si] = max_conf.unsqueeze(-1)

                # use bbox conf if that is the case
                if _bbox_conf_ is not None:
                    bbox_coef = torch.repeat_interleave(
                        _bbox_conf_[..., None], dim=2, repeats=15)

                    rot_loss = L_rot(local_rotmat[:, :, 1:],
                                     local_rotmat_gt[:, :, 1:],
                                     T, conf=bbox_coef)
                else:
                    rot_loss = L_rot(local_rotmat[:, :, 1:],
                                     local_rotmat_gt[:, :, 1:],
                                     T, conf=mano_joint_conf[:, :, 1:])

                loss_dict['rot'] = stg_conf.lambda_rot * rot_loss

            if stg_conf.lambda_reproj > 0:
                reproj_loss = joints2d_loss(
                    joints2d_obs=target['joints2d'], joints2d_pred=joints2d_pred, bbox_conf=_bbox_conf_)
                loss_dict['reproj'] = stg_conf.lambda_reproj * reproj_loss

            if stg_conf.lambda_orient > 0:
                orient_loss = L_orient(
                    output['root_orient'], target['root_orient'], T, bbox_conf=_bbox_conf_)
                loss_dict['orient'] = stg_conf.lambda_orient * orient_loss

            if stg_conf.lambda_trans > 0:
                trans_loss = L_trans(
                    output['trans'], target['trans'], T, bbox_conf=_bbox_conf_)
                loss_dict['trans'] = stg_conf.lambda_trans * trans_loss

            if stg_conf.lambda_rot_smooth > 0:
                rot_smooth_l = rot_smooth_loss(local_rotmat)
                loss_dict['rot_sm'] = stg_conf.lambda_rot_smooth * rot_smooth_l

            if stg_conf.lambda_orient_smooth > 0:
                matrot_root_orient = rotation_6d_to_matrix(root_orient)
                orient_smooth_l = rot_smooth_loss(matrot_root_orient)
                loss_dict['orient_sm'] = stg_conf.lambda_orient_smooth * \
                    orient_smooth_l

            # Smoothness objectives
            if stg_conf.lambda_j3d_smooth > 0:
                joints3d = output['joints3d']

                j3d_smooth_l = pos_smooth_loss(joints3d)
                loss_dict['j3d_sm'] = stg_conf.lambda_j3d_smooth * j3d_smooth_l

            if stg_conf.lambda_trans_smooth > 0:
                # tr = mask_data(output['trans'], mask)
                tr = output['trans']
                tr = tr.reshape(args.nsubject, B, seqlen, 3)
                trans_smooth_l = 0
                for sid in range(args.nsubject):
                    trans_smooth_l += pos_smooth_loss(tr[sid])
                loss_dict['trans_sm'] = stg_conf.lambda_trans_smooth * \
                    trans_smooth_l

            if stg_conf.lambda_motion_prior > 0:

                # if motion_prior_type == "pca":
                #     mp_local_loss = L_PCA(pose)
                # elif motion_prior_type == "gmm":
                #     mp_local_loss = L_GMM(pose)
                # else:
                mp_local_loss = motion_prior_loss(z_l)
                loss_dict['mot_prior'] = stg_conf.lambda_motion_prior * \
                    mp_local_loss

            if stg_conf.lambda_init_z_prior > 0:
                zl_init_prior_l = F.mse_loss(z_l, init_z_l)
                loss_dict['init_z_prior'] = stg_conf.lambda_init_z_prior * \
                    (zl_init_prior_l)

            if hasattr(stg_conf, 'lambda_batch_cs'):
                if stg_conf.lambda_batch_cs > 0:
                    if args.overlap_len == 0:
                        logger.warning(
                            'Batch consistency won\'t be effective since overlap_len is 0')
                    if B > 1:
                        # joints3d = mask_data(output['joints3d'], mask)
                        joints3d = joints3d.reshape(
                            args.nsubject, B, seqlen, -1, 3)
                        batch_cs_l = 0
                        for sid in range(args.nsubject):
                            batch_cs_l += L_pos(
                                joints3d[sid, :-1, -args.overlap_len:], joints3d[sid, 1:, :args.overlap_len], T)
                        loss_dict['batch_cs'] = stg_conf.lambda_batch_cs * batch_cs_l
                    else:
                        if i < 5:
                            logger.warning(
                                'Batch consistency won\'t be effective since batch size is 1')

            if hasattr(stg_conf, 'betas_prior'):
                if stg_conf.betas_prior > 0:
                    if betas is None:
                        logger.error(
                            'Cannot compute betas prior since args.opt_betas is False')
                    betas_prior_l = torch.pow(betas - mean_betas, 2).mean()
                    loss_dict['betas_prior'] = stg_conf.betas_prior * \
                        betas_prior_l

            loss = sum(loss_dict.values())
            loss_dict['loss'] = loss

            # copy loss values to loss_dict_by_step
            for k, v in loss_dict.items():
                loss_dict_by_step[k].append(v.detach().item())

            if not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(opt_params, 5.0)
                optimizer.step()
            else:
                logger.warning('Loss is NaN, skipping this stage')
                return 0

            scheduler.step()
            loss_log_str = f'Stage {stg_id+1} [{i:03d}/{stg_conf.niters}]'
            for k, v in loss_dict.items():
                loss_dict[k] = v.item()
                loss_log_str += f'{k}: {v.item():.3f}\t'
            logger.info(loss_log_str)

        end_time = time.time()

        # save the loss dict.
        joblib.dump(loss_dict_by_step, open(os.path.join(
            args.pkl_output_dir, f'stage_{stg_id}_loss.pkl'), 'wb'))

        print(
            f'Stage {stg_id+1} finished in {time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))}')

        if not betas is None:
            logger.info(f'mean_betas: {mean_betas.detach().cpu().numpy()}')
            logger.info(f'betas: {betas.detach().cpu().numpy()}')

        return z_l, z_g, optim_cam_R, optim_cam_t, trans, root_orient, betas, pose

    def latent_optimization(self, target, z_l, z_g, pose):
        args = self.args
        T = torch.arange(args.data.clip_length)

        cam_R = target['cam_R'].clone()
        cam_t = target['cam_t'].clone()

        optim_trans = target['trans'].clone()
        optim_root_orient = target["root_orient"].clone()

        B, seqlen, _ = optim_trans.shape
        optim_trans.requires_grad = True
        optim_root_orient.requires_grad = True

        if not pose is None:
            optim_pose = pose.clone()
            optim_pose.requires_grad = False
        else:
            optim_pose = pose

        z_global = torch.zeros_like(z_g).to(z_g)
        z_global.requires_grad = False

        init_z_l = z_l.clone().detach()
        init_z_l.requires_grad = False

        # torch.autograd.set_detect_anomaly(True)
        init_cam_orient = torch.eye(3).to(cam_R)
        init_cam_scale = torch.ones(1).to(cam_t) * 2.5

        init_cam_orient = matrix_to_rotation_6d(init_cam_orient)

        full_cam_R = matrix_to_rotation_6d(cam_R.clone())
        full_cam_t = torch.zeros_like(cam_t)

        # take pymafx mean as starting point
        mean_betas = target["betas"].mean(dim=1).mean(dim=0)

        if args.opt_betas:
            betas = torch.autograd.Variable(mean_betas.clone().unsqueeze(
                0).repeat_interleave(args.nsubject, 0), requires_grad=True)
            # logger.info(f'Optimizing betas: {betas}')
        else:
            betas = mean_betas.clone().unsqueeze(0).repeat_interleave(args.nsubject, 0)
            betas.requires_grad = False

        stg_configs = [args.stg1]

        if hasattr(args, 'stg2'):
            stg_configs.append(args.stg2)

        if hasattr(args, 'stg3'):
            stg_configs.append(args.stg3)

        if hasattr(args, 'stg4'):
            stg_configs.append(args.stg4)

        if hasattr(args, 'stg5'):
            stg_configs.append(args.stg5)

        stg_int_results = (init_cam_orient, init_cam_scale, optim_trans, optim_root_orient, z_l, z_global, betas,
                           target, B, seqlen, mean_betas, T, full_cam_R, full_cam_t)

        joblib.dump(stg_int_results, f'{args.pkl_output_dir}/stg_0.pkl')
        logger.info(
            f'Saved intermediate results to {args.pkl_output_dir}/stg_0.pkl')

        is_nan_loss = False
        iter = 0

        # iterate over different optimization steps
        while iter < len(stg_configs):
            stg_conf = stg_configs[iter]
            logger.info(f'Stage {iter+1}: Learning rate: {stg_conf.lr}')
            stg_id = iter

            # break is better than continue here
            if stg_conf.niters == 0 and iter != 0:
                break
            # this corresponds to encode-decode stage, we need to calculate joints2d, joints3d etc.
            elif stg_conf.niters == 0 and iter == 0:
                logger.info('Encode-Decode case')
                # cannot plot loss this case
                self.args.plot_loss = False
                break

            if is_nan_loss:
                # will give error here
                prev_stg_results = joblib.load(
                    f'{self.args.pkl_output_dir}/stg_{stg_id}.pkl')
                _, _, optim_trans, optim_root_orient, z_l, z_global, betas, target, B, seqlen, mean_betas, T, \
                    full_cam_R, full_cam_t = prev_stg_results

            stg_results = self.optim_step(stg_conf, stg_id, z_l, z_global, betas, target,
                                          B, seqlen, optim_trans, optim_root_orient, init_z_l, mean_betas,
                                          T, full_cam_R, full_cam_t, pose=optim_pose)

            if isinstance(stg_results, int):
                is_nan_loss = True
                logger.error(
                    f'[Stage {stg_id+1}] NaN loss detected, restarting stage {stg_id+1}')
                logger.warning(
                    f'Decreasing learning rate by 0.5 for the current stage')
                stg_configs[stg_id].lr *= 0.5

            else:
                z_l, z_global, optim_cam_R, optim_cam_t, optim_trans, optim_root_orient, betas, optim_pose = stg_results

                full_cam_R = matrix_to_rotation_6d(optim_cam_R.detach())
                full_cam_t = optim_cam_t.detach()

                stg_int_results = (init_cam_orient, init_cam_scale, optim_trans, optim_root_orient,
                                   z_l, z_global, betas, target, B, seqlen,
                                   mean_betas, T, full_cam_R, full_cam_t)

                joblib.dump(stg_int_results,
                            f'{self.args.pkl_output_dir}/stg_{stg_id+1}.pkl')
                logger.info(
                    f'Saved intermediate results to {self.args.pkl_output_dir}/stg_{stg_id+1}.pkl')

                iter += 1

        # if args.plot_loss:
        #     plot_list = []

        #     for num in range(iter):
        #         loss_i = joblib.load(f'{args.pkl_output_dir}/stage_{num}_loss.pkl')
        #         plt.figure()

        #         for k, v in loss_i.items():
        #             if not v == []:
        #                 plt.plot(v, label=k)
        #         plt.legend()
        #         plt.title(f'Stage {num}')
        #         plt.savefig(f'{args.pkl_output_dir}/stage_{num}_loss.jpg')

        #         # concatenate all the losses
        #         plot_list.append(cv2.imread(f'{args.pkl_output_dir}/stage_0_loss.jpg'))

        #     plt_concat = np.concatenate(plot_list, axis=0)
        #     cv2.imwrite(f'{args.pkl_output_dir}/all_stages_loss.jpg', plt_concat)

        return z_l, z_global, betas, optim_root_orient, optim_trans, full_cam_R, full_cam_t, optim_pose

    def run(self, input_data: HMPData):
        target = dict()
        # FIXME: confirm on rotation format
        target['pos'] = input_data.pos.to(self.model.device)
        target['rotmat'] = input_data.global_xforms.to(self.model.device)
        target['trans'] = input_data.trans.to(self.model.device)
        target['root_orient'] = input_data.root_orient.to(self.model.device)
        target['cam_R'] = input_data.cam_R.to(self.model.device)
        target['cam_t'] = input_data.cam_t.to(self.model.device)
        target['cam_f'] = input_data.cam_f.to(self.model.device).squeeze(0)

        self.f = target['cam_f'][0, 0].item()
        target['cam_center'] = input_data.cam_center.to(
            self.model.device).squeeze(0)
        target['joints2d'] = input_data.joint2d.to(self.model.device)
        target['joints3d'] = input_data.joint3d.to(self.model.device)
        target['betas'] = input_data.betas.to(self.model.device)
        # target['img_width'] = data['img_width']
        # target['img_height'] = data['img_height']
        # target['img_dir'] = data['img_dir']
        # target['frame_id'] = gt_frame_id
        # target['save_path'] = data['save_path']
        # target['rh_verts'] = rh_verts   # for debug purposes
        # target['config_type'] = data['config_type']
        # target['handedness'] = gt_values['handedness']

        # refre: applicatgion.py:load_aist_data()
        data = {
            # for encode_local
            "pos": target["pos"],
            "velocity": None,
            "global_xform": input_data.global_xforms,  # (T, J, 6)
            "angular": None,
            # for encode_global
            "root_orient": target["root_orient"],  # (T, 6)
            "root_vel": None,
        }
        # follow amass.py:get_stage2_res function

        # (T, J, 3, 3)
        data['angular'] = estimate_angular_velocity(
            rotation_6d_to_matrix(data['global_xform']), dt=1.0 / self.args.data.fps).squeeze(0)
        data['velocity'] = estimate_linear_velocity(
            data["pos"], dt=1.0 / self.args.data.fps).squeeze(0)
        data['root_vel'] = estimate_linear_velocity(
            target["trans"], dt=1.0 / self.args.data.fps).squeeze(0)

        self.model.set_input(data)
        z_l, _, _ = self.model.encode_local()
        z_g, _, _ = self.model.encode_global()

        z_l, z_g, opt_betas, opt_root_orient, opt_trans, cam_R, cam_t, _ = self.latent_optimization(
            target, z_l, z_g, pose=None)

        with torch.no_grad():
            step = 1.0
            B, seqlen, _ = opt_trans.shape
            output = self.model.decode(
                z_l, z_g=z_g, length=self.args.data.clip_length, step=step)
            output['betas'] = opt_betas[:, None,
                                        None, :].repeat(1, B, seqlen, 1)
            output['trans'] = opt_trans.clone().reshape(
                B * self.args.nsubject, seqlen, 3)

            # notice, the value in original output['root_orient'] is very different from opt_root_orient
            output['root_orient'] = opt_root_orient

            rh_mano_out = forward_mano(output, self.fk, self.hand_model)
            joints3d_pred = rh_mano_out.joints.view(B, seqlen, -1, 3)
            vertices_pred = rh_mano_out.vertices.view(B, seqlen, -1, 3)

            optim_cam_R = rotation_6d_to_matrix(cam_R)
            optim_cam_t = cam_t

            joints2d_pred = get_joints2d(joints3d_pred=joints3d_pred,
                                         cam_t=optim_cam_t.unsqueeze(
                                             0).repeat_interleave(self.args.nsubject, 0),
                                         cam_R=optim_cam_R.unsqueeze(
                                             0).repeat_interleave(self.args.nsubject, 0),
                                         cam_f=torch.tensor([self.f, self.f]),
                                         cam_center=target['cam_center'])
        return output, joints3d_pred, vertices_pred, joints2d_pred, optim_cam_R, optim_cam_t


if __name__ == "__main__":
    hmp = HMPModel(f=None)
    input_data = load_hot3d_folder("../hot3d_clips_data/clip-001849")
    output, joints3d_pred, vertices_pred, joints2d_pred, optim_cam_R, optim_cam_t = hmp.run(
        input_data)


# args = Arguments(
#     './configs', filename='in_the_wild_sample_config.yaml')
# args.dataname = "in_the_wild"
# ngpu = 1
# model = Architecture(args, ngpu)
# model.load_model(optimal=True)
# model.eval()
# # setup input data (hand, camera)
# target = ...


# data = {
#     # for encode_local
#     "pos": None,
#     "veclocity": None,
#     "global_xforms": None,
#     "angular": None,
#     # for encode_global
#     "root_orient": None,
#     "root_vel": None,
# }

# model.set_input(data)
# step = 1.0
# # FIXME: this encode step relies on model.input_data is setup beforehand
# # for now, it will make use of data setup in the previous cell
# z_l, _, _ = model.encode_local()
# z_g, _, _ = model.encode_global()

# # FIXME: explicitly write down what is needed in target
# # optimize pose directly
# # pose_aa = matrix_to_axis_angle(model.input_data["rotmat"].detach(
# # ).clone()) if motion_prior_type in ['gmm', 'pca'] else None

# pose_aa = None

# z_l, z_g, opt_betas, opt_root_orient, opt_trans, cam_R, cam_t, opt_pose_aa = latent_optimization(
#     target, T=T, z_l=z_l, z_g=z_g, pose=pose_aa)
# output = model.decode(
#     z_l, z_g=z_g, length=args.data.clip_length, step=step)
# # out of box after decode, it has keys: ['rotmat', 'pos', 'root_orient']
# # (2, 128, 16, 3, 3); (2, 128, 16, 3); (2, 128, 6)

# # opt_beta, opt_root_orient, opt_trans, cam_R, cam_t
# # (1, 10); (2, 128, 6); (2, 128, 3); (2, 128, 6); (2, 128, 3)

# output['betas'] = opt_betas[:, None, None, :].repeat(1, B, seqlen, 1)
# output['trans'] = opt_trans.clone().reshape(B*args.nsubject, seqlen, 3)

# # notice, the value in original output['root_orient'] is very different from opt_root_orient
# output['root_orient'] = opt_root_orient

# rh_mano_out = forward_mano(output)
# joints3d_pred = rh_mano_out.joints.view(B, seqlen, -1, 3)
# vertices_pred = rh_mano_out.vertices.view(B, seqlen, -1, 3)

# optim_cam_R = rotation_6d_to_matrix(cam_R)
# optim_cam_t = cam_t

# joints2d_pred = get_joints2d(joints3d_pred=joints3d_pred,
#                              cam_t=optim_cam_t.unsqueeze(
#                                  0).repeat_interleave(args.nsubject, 0),
#                              cam_R=optim_cam_R.unsqueeze(
#                                  0).repeat_interleave(args.nsubject, 0),
#                              cam_f=torch.tensor([5000., 5000.]),
#                              cam_center=target['cam_center'])


# decode form latent space to hand pose

# steps = [1.0]
# for step in steps:
#     with torch.no_grad():
#         B, seqlen, _ = opt_trans.shape

#         # if motion_prior_type in ["pca", "gmm"]:
#         #     output = {"rotmat": axis_angle_to_matrix(opt_pose_aa)}
#         # else:
#         #     output = model.decode(
#         #         z_l, z_g=z_g, length=args.data.clip_length, step=step)

#         output = model.decode(
#             z_l, z_g=z_g, length=args.data.clip_length, step=step)

#         output['betas'] = opt_betas[:, None, None, :].repeat(1, B, seqlen, 1)
#         output['trans'] = opt_trans.clone().reshape(B*args.nsubject, seqlen, 3)
#         output['root_orient'] = opt_root_orient

#         rh_mano_out = forward_mano(output)
#         joints3d_pred = rh_mano_out.joints.view(B, seqlen, -1, 3)
#         vertices_pred = rh_mano_out.vertices.view(B, seqlen, -1, 3)

#         optim_cam_R = rotation_6d_to_matrix(cam_R)
#         optim_cam_t = cam_t

#         joints2d_pred = get_joints2d(joints3d_pred=joints3d_pred,
#                                      cam_t=optim_cam_t.unsqueeze(
#                                          0).repeat_interleave(args.nsubject, 0),
#                                      cam_R=optim_cam_R.unsqueeze(
#                                          0).repeat_interleave(args.nsubject, 0),
#                                      cam_f=torch.tensor([5000., 5000.]),
#                                      cam_center=target['cam_center'])

#     fps = int(args.data.fps / step)
#     # criterion_geo = GeodesicLoss()

#     rotmat = output['rotmat']  # (B, T, J, 3, 3)
#     rotmat_gt = target['rotmat']  # (B, T, J, 3, 3)
#     b_size, _, n_joints = rotmat.shape[:3]

#     local_rotmat = fk.global_to_local(
#         rotmat.view(-1, n_joints, 3, 3))  # (B x T, J, 3, 3)
#     local_rotmat = local_rotmat.view(b_size, -1, n_joints, 3, 3)
#     local_rotmat_gt = fk.global_to_local(
#         rotmat_gt.view(-1, n_joints, 3, 3))  # (B x T, J, 3, 3)
#     local_rotmat_gt = local_rotmat_gt.view(b_size, -1, n_joints, 3, 3)

#     root_orient = rotation_6d_to_matrix(opt_root_orient)  # (B, T, 3, 3)
#     root_orient_gt = rotation_6d_to_matrix(
#         target['root_orient'])  # (B, T, 3, 3)

#     global_trans = opt_trans.clone()
#     B = global_trans.shape[0]

#     if args.data.root_transform:
#         local_rotmat[:, :, 0] = root_orient
#         local_rotmat_gt[:, :, 0] = root_orient_gt

#     trans = global_trans
#     trans_gt = target['trans']  # (B, T, 3)

#     # concatenate results in a certain way (what for?)
#     if trans.shape[0] > 1:

#         # batch optimization
#         if args.overlap_len == 0:
#             # concat_results
#             trans = trans.reshape(1, -1, 3)
#             trans_gt = trans_gt.reshape(1, -1, 3)
#             local_rotmat = local_rotmat.reshape(1, -1, n_joints, 3, 3)
#             local_rotmat_gt = local_rotmat_gt.reshape(1, -1, n_joints, 3, 3)
#             cam_R = target['cam_R'].reshape(1, -1, 3, 3)
#             cam_t = target['cam_t'].reshape(1, -1, 3)
#             joints2d_pred = joints2d_pred.reshape(
#                 1, -1, *joints2d_pred.shape[2:])
#             joints3d_pred = joints3d_pred.reshape(
#                 1, -1, *joints3d_pred.shape[2:])
#             vertices_pred = vertices_pred.reshape(
#                 1, -1, *vertices_pred.shape[2:])

#             for k, v in target.items():
#                 if not (k in ['cam_f', 'cam_center', 'img_height', 'img_width', 'img_dir', 'save_path', 'frame_id', 'config_type', 'rh_verts', 'handedness']):

#                     target[k] = v.reshape(1, -1, * v.shape[2:])

#         else:
#             B, T, J, _, _ = output["rotmat"].shape

#             def resh(x): return x.reshape(1, B, *x.shape[1:])

#             def concat(x):
#                 x = resh(x)
#                 ls = [x[:, 0]]
#                 for bid in range(1, x.shape[1]):
#                     ls.append(x[:, bid, args.overlap_len:])
#                 return torch.cat(ls, dim=1)[:, :args.orig_seq_len]

#             # concat_cam = lambda x: torch.cat([x[0]] + [y[args.overlap_len:] for y in x[1:]], dim=0).unsqueeze(0)[:, :args.orig_seq_len]
#             trans = concat(trans)
#             trans_gt = concat(trans_gt)
#             local_rotmat = concat(local_rotmat)
#             local_rotmat_gt = concat(local_rotmat_gt)
#             cam_R = target['cam_R'].reshape(1, -1, 3, 3)
#             cam_t = target['cam_t'].reshape(1, -1, 3)
#             joints2d_pred = concat(joints2d_pred)
#             joints3d_pred = concat(joints3d_pred)
#             vertices_pred = concat(vertices_pred)

#             for k, v in target.items():
#                 if not (k in IGNORE_KEYS):
#                     target[k] = concat(v)

# dump results
# for i in range(local_rotmat.shape[0]): # (1, T. J, 3, 3)

#     poses = c2c(matrix_to_axis_angle(local_rotmat[i]))  # (T, J, 3)
#     poses_gt = c2c(matrix_to_axis_angle(local_rotmat_gt[i]))  # (T, J, 3)

#     poses = poses.reshape((poses.shape[0], -1))  # (T, 48)
#     poses_gt = poses_gt.reshape((poses_gt.shape[0], -1))  # (T, 66)

#     limit = poses.shape[0]

#     pred_save_path = os.path.join(output_dir, f'recon_{offset + i:03d}_{fps}fps.npz')
#     gt_save_path = pred_save_path if args.raw_config else os.path.join(output_dir, f'recon_{offset + i:03d}_gt.npz')

#     # save opt results if not _pymafx_raw or _metro_raw
#     if not args.raw_config:
#         np.savez(pred_save_path,
#                 poses=poses[:limit],
#                 trans=c2c(trans[i][:limit]),
#                 betas=opt_betas.detach().cpu().numpy()[:limit],
#                 gender=args.data.gender,
#                 mocap_framerate=fps,
#                 cam_R=c2c(target['cam_R'][i][0]),
#                 cam_t=c2c(target["cam_t"][i][0]),
#                 cam_f=c2c(target['cam_f'].unsqueeze(0)),
#                 cam_center=c2c(target['cam_center'].unsqueeze(0)),
#                 joints_2d=c2c(joints2d_pred[i][:limit]),
#                 keypoints_2d=c2c(target['joints2d'][i][:limit]),  # detected keypoints with confidence
#                 vertices=c2c(vertices_pred[i][:limit]),
#                 save_path=pred_save_path,
#                 handedness=target["handedness"],
#                 img_dir=str(target['img_dir']),
#                 frame_id=target['frame_id'],
#                 img_height=target['img_height'],
#                 img_width=target['img_width'],
#                 config_type=target['config_type'],
#                 joints_3d=c2c(joints3d_pred)[i][:limit])

#     if ("_encode_decode" in gt_save_path) or args.raw_config:
#         np.savez(gt_save_path,
#                 poses=poses_gt[:limit],
#                 trans=c2c(trans_gt[i][:limit]),
#                 betas=target["betas"][0].detach().cpu().numpy()[:limit],
#                 vertices=target["rh_verts"].detach().cpu().numpy()[:limit],
#                 config_type=target['config_type'] if args.raw_config else None,
#                 gender=args.data.gender,
#                 save_path=gt_save_path,
#                 mocap_framerate=args.data.fps,
#                 frame_id=target['frame_id'],
#                 img_height=target['img_height'],
#                 img_width=target['img_width'],
#                 cam_R=c2c(target['cam_R'][i][0]),
#                 cam_t=c2c(target['cam_t'][i][0]),
#                 cam_f=c2c(target['cam_f'].unsqueeze(0)),
#                 cam_center=c2c(target['cam_center'].unsqueeze(0)),
#                 joints2d=c2c(target['joints2d'][i, ...,])[:limit],
#                 joints_2d=c2c(target['joints2d'][i, ...,])[:limit],
#                 keypoints_2d=c2c(target['joints2d'][i, ...,])[:limit], # duplicated, because alignment reads that key value
#                 joints_3d=c2c(target['joints3d'][i])[:limit])
