1. nohup python tools/train.py --cfg_file cfgs/lightstereo/lightstereo_m_kitti.yaml > train.log 2>&1 &


2. python tools/eval.py --cfg_file "/home/aniruth/Desktop/Courses/Independent - Study/OpenStereoSLAM/cfgs/lightstereo/lightstereo_m_kitti.yaml" --eval_data_cfg_file "/home/aniruth/Desktop/Courses/Independent - Study/OpenStereoSLAM/cfgs/lightstereo/kitti12_eval.yaml" --pretrained_model "/home/aniruth/Desktop/Courses/Independent - Study/OpenStereoSLAM/output/KittiDataset/LightStereo/lightstereo_m_kitti/default-baseline/ckpt/checkpoint_best.pth"

python tools/eval.py --cfg_file "/home/aniruth/Desktop/Courses/Independent - Study/OpenStereoSLAM/cfgs/lightstereo/lightstereo_m_kitti.yaml" --eval_data_cfg_file "/home/aniruth/Desktop/Courses/Independent - Study/OpenStereoSLAM/cfgs/lightstereo/kitti12_eval.yaml" --pretrained_model "/home/aniruth/Desktop/Courses/Independent - Study/OpenStereoSLAM/output/KittiDataset/LightStereo/lightstereo_m_kitti/default-baseline/ckpt/checkpoint_best.pth"

Baseline: 
Evaluation metrics: {'d1_all': tensor(1.6970), 'epe': tensor(0.5399), 'thres_1': tensor(10.1487), 'thres_2': tensor(3.6810), 'thres_3': tensor(2.0529)}
Quantized : 
Evaluation metrics: {'d1_all': tensor(1.1602), 'epe': tensor(0.4348), 'thres_1': tensor(7.0816), 'thres_2': tensor(2.4537), 'thres_3': tensor(1.3806)}

