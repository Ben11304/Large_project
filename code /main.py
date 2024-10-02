import utilis


root_path="/Users/mac/Dev/Large_project/Large_project/data/tiny_coco_dataset/tiny_coco/train2017"
target="/Users/mac/Dev/Large_project/Large_project/Target"
model_name="yolov10x"
device="mps"
label, group_label= utilis.read_json("label.json")
utilis.main(root_path,target,model_name,device, label, group_label)