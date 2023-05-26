from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO('yolov8n-seg.pt')

img = cv2.imread('reference.jpg')
print('Image shape: ', img.shape) # 720, 1280, 3

X_OFFSET = 1000
Y_OFFSET = 150
PEOPLE_CLS = 0

img_reference = img#[Y_OFFSET:,:X_OFFSET,:]

video_path = 'video-survillance.mp4'

def mask_to_shadow(image, mask, reference, shadow=True):
    mask =( cv2.merge([mask,mask,mask])*255).astype(np.uint8)
    mask_inverse = cv2.bitwise_not(mask)
    # print(image.shape, mask_inverse.shape, image.dtype, mask_inverse.dtype)
    mask_image = cv2.bitwise_and(image, mask_inverse)

    mask_reference = cv2.bitwise_and(reference, mask)

    if shadow:
        # red_mask = mask.copy()
        mask[:,:,:2]= 0
        mask_reference = cv2.addWeighted(mask_reference, 0.5, mask, 0.5, 0)

    shadow_output = cv2.bitwise_or(mask_image, mask_reference)

    return shadow_output

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print('Error opening video file')

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # cv2.imshow('Frame', frame)

        frame = frame#[Y_OFFSET:,:X_OFFSET,:]
        # cv2.imwrite('reference.jpg', frame)

        results = model(frame)
        cv2.imshow('yolo', results[0].plot())
        cls = results[0].boxes.cls

        pos = (cls==PEOPLE_CLS).nonzero().squeeze().tolist()

        mask = np.zeros((frame.shape[0],frame.shape[1]))

        # print(cls, pos)
        if pos or pos==0:
            print(cls, pos)
            try: 
                m = results[0].masks.data.cpu().numpy()[pos]
                mask = cv2.resize(m, (frame.shape[1],frame.shape[0]))
            except:
                print('multiple persons')
                for p in pos:
                    m = results[0].masks.data.cpu().numpy()[p]
                    # print(m.shape, mask.shape)
                    mask += cv2.resize(m, (frame.shape[1],frame.shape[0]))

        # print(frame.shape, mask.shape, img_reference.shape)
        shadow_img = mask_to_shadow(frame, mask, img_reference)
        cv2.imshow('Ghost', shadow_img)
        cv2.imshow('frame', mask)

        if cv2.waitKey(25) & 0xFF==ord('q'):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()
