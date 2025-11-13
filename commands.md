1. nohup python tools/train.py --cfg_file cfgs/lightstereo/lightstereo_m_kitti.yaml > train.log 2>&1 &


2. python tools/eval.py --cfg_file "/home/aniruth/Desktop/Courses/Independent - Study/OpenStereoSLAM/cfgs/lightstereo/lightstereo_m_kitti.yaml" --eval_data_cfg_file "/home/aniruth/Desktop/Courses/Independent - Study/OpenStereoSLAM/cfgs/lightstereo/kitti12_eval.yaml" --pretrained_model "/home/aniruth/Desktop/Courses/Independent - Study/OpenStereoSLAM/output/KittiDataset/LightStereo/lightstereo_m_kitti/default/ckpt/checkpoint_best.pth"

