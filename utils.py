import cv2
import numpy as np
from ultralytics import YOLO

def mask_to_shadow(image: np.ndarray, mask: np.ndarray, 
                   reference: np.ndarray, shadow:bool = True) -> np.ndarray:
    '''
    Args:
    original image:
    mask: segmentation prediction only in person label
    the reference image: image to cover the mask
    shadow: shadow effect
    
    Return the original image with a layer privacy (red layer), the layer makes 
    people shadows based on the provided mask
    '''
    mask = (cv2.merge([mask,mask,mask])*255).astype(np.uint8)
    mask_inverse = cv2.bitwise_not(mask)

    mask_image = cv2.bitwise_and(image, mask_inverse)
    mask_reference = cv2.bitwise_and(reference, mask)

    if shadow:
        mask[:,:,:2]= 0
        mask_reference = cv2.addWeighted(mask_reference, 0.5, mask, 0.5, 0)

    return cv2.bitwise_or(mask_image, mask_reference)


class YoloSeg:
    def __init__(self, model_wights, cls):
        self.model = YOLO(model=model_wights)
        self.cls = cls # person label idx

    def processing(self, output)-> np.ndarray:
        
        data = output[0]
        H, W = data.orig_shape

        # Extract found labels and position 
        cls = data.boxes.cls
        pos = (cls==self.cls).nonzero().squeeze().tolist()

        # Initiate empty mask 
        mask = np.zeros((H, W))

        if pos or pos==0:
            # Extract masks from person label
            try: 
                m = data.masks.data.cpu().numpy()[pos]
                mask = cv2.resize(m, (W,H))
            except:
                print('multiple detections')
                for p in pos:
                    m = data.masks.data.cpu().numpy()[p]
                    mask += cv2.resize(m, (W,H))
        
        return mask

    def predict(self, img: np.ndarray)-> np.ndarray:
        '''
        Return mask that belongs only to a specific label
        '''
        output = self.model(img)
        mask = self.processing(output=output)
        return mask




if __name__ == '__main__':
    PERSON_IDX = 0
    model = YoloSeg(model_wights='yolov8n-seg.pt', cls=PERSON_IDX)