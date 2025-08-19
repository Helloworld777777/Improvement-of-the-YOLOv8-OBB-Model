from ultralytics import YOLO


def main():
    model = YOLO('yolov8n-obb.yaml') # build from YAML and transfer weights
    model.train(data='dota8-obb.yaml', epochs=500, imgsz=1024, batch=24)
if __name__ == '__main__':
    main()
