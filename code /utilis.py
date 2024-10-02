from ultralytics import YOLO
import os
import cv2
import numpy as np
import json
from tqdm import tqdm
import shutil


def read_json(json_path):
    # Đọc dữ liệu từ file JSON
    with open(json_path, "r") as file:
        data = json.load(file)

    # Lưu vào labels và group_label từ dữ liệu JSON
    labels = data['data']['labels']
    group_label = data['data']['group_label']
    return labels, group_label

def get_all_image_files(root_folder):
    # Các định dạng ảnh thường gặp
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff','.JPG'}
    image_files = []
    
    # Sử dụng os.walk để đệ quy qua tất cả các thư mục và file
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            # Kiểm tra xem file có phải là file ảnh không
            if os.path.splitext(filename)[1].lower() in image_extensions:
                image_files.append(os.path.join(dirpath, filename))
    return image_files

def predict(chosen_model, img, classes=[], conf=0.5):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf, verbose=False)
    else:
        results = chosen_model.predict(img, conf=conf, verbose=False)

    return results

def predict_and_detect(chosen_model, img, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1):
    results = predict(chosen_model, img, classes, conf=conf)
    for result in results:
        for box in result.boxes:
            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), rectangle_thickness)
            cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), text_thickness)
    return img, results
def find_group_for_label(label_name,group_label_dict):
    for group, labels in group_label_dict.items():
        if label_name in labels:
            return group
    return 'Unknown'


def sort_bb_group(results, top, group_label):
    areas_bb_with_labels = []
    if not results or len(results) == 0:
        return areas_bb_with_labels
    labels_name=results[0].names
    labels_name = list(labels_name.values())
    labels = results.boxes.cls
    bbs = results.boxes.xyxy
    for bb, label in zip(bbs, labels):
        x, y, x_, y_ = bb
        area = (x_ - x) * (y_ - y)  
        area = int(area.item()) 
        areas_bb_with_labels.append((area, int(label.item()))) 
    areas_bb_with_labels = sorted(areas_bb_with_labels, key=lambda x: x[0], reverse=True)
    top_areas_bb_with_labels = areas_bb_with_labels[:top]
    top_areas_with_group_labels = []
    for area, label in top_areas_bb_with_labels:
        label_name = labels_name[label] 
        group = find_group_for_label(label_name,group_label)
        top_areas_with_group_labels.append((area, label_name, group))
    return top_areas_with_group_labels

def calculate_total_area_by_group(output):
    group_area_totals = {}
    for area, label, group in output:
        if group in group_area_totals:
            group_area_totals[group] += area
        else:
            group_area_totals[group] = area 

    return group_area_totals

def calculate_total_area_by_group(output):
    group_area_totals = {}
    for area, label, group in output:
        if group in group_area_totals:
            group_area_totals[group] += area
        else:
            group_area_totals[group] = area 

    return group_area_totals

def get_label(group_area_totals):
    first_key = next(iter(group_area_totals), "Tools & Other")
    return first_key

def redefine_category(results):
    names_dict = results.names
    labels = list(names_dict.values())
    grouped_labels = {
        'Person': ['person'],
        'Vehicle': ['bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat'],
        'Traffic Sign': ['traffic light', 'stop sign', 'fire hydrant', 'parking meter'],
        'Home': ['bench', 'chair', 'couch', 'bed', 'dining table', 'toilet', 'microwave', 'oven', 'refrigerator', 'sink', 'toaster', 'potted plant', 'clock', 'vase'],
        'Animals': ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'],
        'Food': ['banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake'],
        'Accessories': ['backpack', 'handbag', 'umbrella', 'tie', 'suitcase'],
        'Sports': ['frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket'],
        'Kitchen': ['bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl'],
        'Electronics': ['tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone'],
        'Tools & Other': ['scissors', 'book', 'teddy bear', 'hair drier', 'toothbrush']
    }
    import json
    # Cấu trúc dữ liệu JSON
    data = {
        "data": {
            "labels": labels,
            "group_label": grouped_labels
        }
    }

    # Lưu vào file JSON
    with open("label.json", "w") as file:
        json.dump(data, file, indent=4)

    print("Đã lưu dữ liệu vào file labels_grouped.json")



def main(root_path, target, model_name, device, label,group_label):
    model = YOLO(model_name).to(device)  # Tải model
    list_img = get_all_image_files(root_path)  # Lấy danh sách tất cả các file ảnh

    # Thêm tqdm để hiển thị tiến trình của quá trình lặp qua các ảnh
    for img_path in tqdm(list_img, desc="Processing Images"):
        img = cv2.imread(img_path)
        img, result = predict_and_detect(model, img, [], 0.6)  # Dự đoán và phát hiện
        ar = sort_bb_group(result[0], 6, group_label)  # Sắp xếp bounding box
        re = calculate_total_area_by_group(ar)  # Tính tổng diện tích theo group
        group = next(iter(re), "no_label")  # Lấy group đầu tiên hoặc "no_label" nếu không có
        if not os.path.exists(f"{target}"):
            os.mkdir(f"{target}")
        if not os.path.exists(f"{target}/{group}"):
            os.mkdir(f"{target}/{group}")  # Create the destination folder
        img_name = img_path.split("/")[-1]  # Lấy tên file ảnh
        destination_path = f"{target}/{group}/{img_name}"  # Đường dẫn đích
        shutil.copyfile(img_path, destination_path)  # Sao chép ảnh đến đích

    print("Processing completed.")

    
    
