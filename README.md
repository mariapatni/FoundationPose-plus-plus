# FoundationPose++

## Environment Setup
check [install.md](./Install.md) to install all the dependencies

## Prepare your testcase data
Your testcase data should be formatted like:
```
$PROJECT_ROOT/testcase
└── color
    ├── 0.jpg
    ├── 1.jpg
    └── ...
└── depth
    ├── 0.png
    ├── 1.png
    └── ...
└── mesh
    ├── 1x4.stl
```
There should be an RGB image file and a corresponding depth file for each frame, as well as a mesh file of the object, according to [FoundationPose](https://github.com/NVlabs/FoundationPose) data format.

## Run webapi servers (QwenVL and SAM)
`cd $PROJECT_ROOT`

Fill in the path of the downloaded weights in `run_servers.sh`.

Run all the servers using `bash run_servers.sh`

## Get the object mask of the first frame to initialize the 2D tracker
 `cd $PROJECT_ROOT`

Run the following script to get the position of the bounding box.
```
BOUNDING_BOX_POSITION=$(python src/utils/obj_bbox.py \
    --frame_path $PROJECT_ROOT/testcase/color/0.jpg \
    --object_name $DESCRIPTION_OF_THE_OBJECT \
    --reference_img_path $PATH_OF_REFERENCE_IMAGE)
```

Then run the following script to get the mask of object in the first frame.
```
python src/utils/obj_mask.py  \
    --frame_path $PROJECT_ROOT/testcase/color/0.jpg \
    --bbox_xywh $BOUNDING_BOX_POSITION \
    --output_mask_path $PROJECT_ROOT/0_mask.jpg
```

`$DESCRIPTION_OF_THE_OBJECT`: the description of an object to help QwenVL anchor box positions, better in Chinese.

`$PATH_OF_REFERENCE_IMAGE`: you can provide what the object looks like to help QwenVL anchor box positions more precisely.

## Track Inference
 `cd $PROJECT_ROOT`

```
python src/obj_pose_track.py \
--rgb_seq_path $PROJECT_ROOT/test_case/color \
--depth_seq_path $PROJECT_ROOT/test_case/depth \
--mesh_path $PROJECT_ROOT/test_case/depth/1x4.stl \
--init_mask_path $PROJECT_ROOT/0_mask.jpg \
--pose_output_path $PROJECT_ROOT/pose \
--mask_visualization_path $PROJECT_ROOT/mask_visualization \
--bbox_visualization_path $PROJECT_ROOT/bbox_visualization \
--pose_visualization_path $PROJECT_ROOT/pose_visualization \
--activate_2d_tracker True \
--activate_kalman_filter True
```

You can deactivate 2d_tracker or kalman filter according to your use case.