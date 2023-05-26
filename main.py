from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO('yolov8n-seg.pt')

img = cv2.imread('reference.jpg')
print('Image shape: ', img.shape) # 720, 1280, 3

X_OFFSET = 1000
Y_OFFSET = 150
PEOPLE_CLS = 0

img_reference = img[Y_OFFSET:,:X_OFFSET,:]

video_path = 'video-survillance.mp4'

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print('Error opening video file')

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # cv2.imshow('Frame', frame)

        frame = frame[Y_OFFSET:,:X_OFFSET,:]
        # cv2.imwrite('reference.jpg', frame)

        results = model(frame)
        cv2.imshow('yolo', results[0].plot())
        cls = results[0].boxes.cls

        pos = (cls==PEOPLE_CLS).nonzero().squeeze().tolist()

        mask = np.zeros((frame.shape[0],frame.shape[1]))

        print(cls, pos)
        if pos or pos==0:
            # print(cls, pos)
            try: 
                mask = results[0].masks.data.cpu().numpy()[pos]
                mask = cv2.resize(mask, (frame.shape[1],frame.shape[0]))
            except:
                print('multiple persons')
                continue

        cv2.imshow('frame', mask)

        if cv2.waitKey(25) & 0xFF==ord('q'):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()

# results = model('reference.jpg')

# res_plotted = results[0].plot()
# cv2.imshow("result", res_plotted)

# Y = 170

# cv2.waitKey(0)
# cv2.destroyAllWindows()