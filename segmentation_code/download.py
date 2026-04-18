
from roboflow import Roboflow
rf = Roboflow(api_key="8431sjMG859LoZZYdahe")
project = rf.workspace("aashma-workspace").project("segmentappledisease")
version = project.version(5)
dataset = version.download("coco-segmentation")



"""
uv run main.py \
  --dataset-root ./SegmentAppledisease-5 \
  --model yolo11x-seg.pt \
  --imgsz 640 \
  --epochs 30 \
  --batch 16 \
  --device 5 \
  --project ./runs/segment/apple_disease_seg \
  --name yolo11x_seg_v5_adamw_30 \
  --workers 2 \
  --patience 10
  
"""
