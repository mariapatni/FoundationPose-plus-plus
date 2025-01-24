# 调用qwen-vl，得到首帧bbox
python src/utils/obj_bbox.py

# 调用sam，得到首帧mask
python src/utils/obj_mask.py  --frame_path /workspace/yanwenhao/detection/test_case2/color/0.jpg --bbox_xywh "[632, 419, 198, 59]" --output_mask_path /workspace/yanwenhao/detection/FoundationPose++/masks/0_m.jpg

python src/obj_pose_track.py  --pose_output_path /workspace/yanwenhao/detection/FoundationPose++/output_pose 