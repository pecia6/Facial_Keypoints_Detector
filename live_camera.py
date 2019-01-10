
import os
import sys
import cv2
import torch
from models import Net
import numpy as np
from torch.autograd import Variable
import numpy as np
import cv2

cap = cv2.VideoCapture(0)


def main():
    
    net = Net()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net.to(device)
    net.load_state_dict(torch.load('saved_models/keypoints_model_0947.pt'))

    while(True):
        # Capture frame-by-frame
        ret, image_1 = cap.read()

        # Frame transformation
        gray_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)

        # Detect faces
        face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')
        faces_1 = face_cascade.detectMultiScale(gray_1, 1.1, 5)

        # make a copy of the original image to plot detections on
        image_with_detections_1 = image_1.copy()

        # loop over the detected faces, mark the image where each face is found
        for (x, y, w, h) in faces_1:
            # face = gray_1
            roi = gray_1[y:y + int(h*1.2), x:x + int(w*1.2)]
            org_shape = roi.shape
            roi = roi / 255.0
            roi = cv2.resize(roi, (224, 224))
            roi = roi.reshape(roi.shape[0], roi.shape[1], 1)
            roi = np.transpose(roi, (2, 0, 1))
            roi = torch.from_numpy(roi)
            roi = Variable(roi)
            roi = roi.type(torch.cuda.FloatTensor)
            roi = roi.unsqueeze(0)
            predicted_key_pts = net(roi)
            predicted_key_pts = predicted_key_pts.view(68, -1)
            predicted_key_pts = predicted_key_pts.data
            predicted_key_pts = predicted_key_pts.cpu().numpy()
            predicted_key_pts = predicted_key_pts * 50.0 + 100

            predicted_key_pts[:, 0] = predicted_key_pts[:, 0] * org_shape[0] / 224 + x
            predicted_key_pts[:, 1] = predicted_key_pts[:, 1] * org_shape[1] / 224 + y

            for (x_point, y_point) in zip(predicted_key_pts[:, 0], predicted_key_pts[:, 1]):
                cv2.circle(image_with_detections_1, (x_point, y_point), 3, (0, 255, 0), -1)
       
        cv2.imshow('Color', image_with_detections_1)
   
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    return 0



if __name__ == '__main__':
    print(__doc__)

    main()