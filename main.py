import cv2
from utils import mask_to_shadow, YoloSeg

def main(video_path: str) -> None:

    # 1. Parameters: Area of interest
    X_OFFSET = 1000
    Y_OFFSET = 150

    PERSON_IDX = 0

    # 2. Reference image

    img = cv2.imread('reference.jpg')
    img_reference = img[Y_OFFSET:,:X_OFFSET,:]

    # 3. Initiate model
    model = YoloSeg(model_wights='yolov8n-seg.pt', cls=PERSON_IDX)

    # 4. Video Loop
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print('Error opening video file')

    while cap.isOpened():
        
        ret, frame_c = cap.read()
        if ret:

            # A. area of interest
            frame = frame_c[Y_OFFSET:,:X_OFFSET,:]

            # B. mask prediction
            mask = model.predict(frame)

            # C. mask -> shadow layer -> full frame 
            mask = mask_to_shadow(image=frame, mask=mask, reference=img_reference)
            frame_c[Y_OFFSET:,:X_OFFSET,:] = mask

            # D. Visualization
            cv2.rectangle(frame_c, (0, Y_OFFSET),(X_OFFSET,img.shape[0]), color=(255,0,0),thickness=2)
            cv2.imshow('Layer in Area', frame_c)

            if cv2.waitKey(25) & 0xFF==ord('q'):
                break

        else:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    video_path = 'video-survillance.mp4'
    main(video_path)
