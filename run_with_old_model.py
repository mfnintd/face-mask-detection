import cv2 as cv
import numpy as np
from tensorflow import keras
import mtcnn
model = keras.models.load_model('./vgg16_model_mask_detection_epochs_10.h5')
mt = mtcnn.MTCNN() #Thư viện MTCNN nhận diện gương mặt

vid = cv.VideoCapture(0) #camera số 0
while (True):
    _, frame = vid.read() #frame: ảnh thu vào từ camera
    frame = cv.flip(frame, 1) #lật ảnh
    #frame_cvt = frame
    frame_cvt = cv.cvtColor(frame, cv.COLOR_BGR2RGB) #chuyển từ BGR sang RGB để xử lý ảnh
    #frame_resize = cv.resize(frame_cvt, (224, 224))
    #frame /= 255

    faces = mt.detect_faces(frame_cvt) # Trích ra tất cả các gương mặt có trong ảnh: vị trí góc trái trên
    for face in faces:
        x1, y1, width, height = face['box']
        x2, y2 = x1 + width, y1 + height #lấy điểm phải dưới
        face = frame_cvt[y1 : y2 , x1 : x2]
        face_resize = cv.resize(face, (224, 224))
        result = model.predict(face_resize.reshape(1, 224, 224, 3))[0][0] #kết quả từ 0 đến 1 tượng trưng cho đeo khẩu trang hay không
        
        color, content = ((0, 255, 0), "An toan") if result < 0.5 else ((0, 0, 255), "De nghi deo khau trang")

        #print(result)
        ##
        
        #result = 0 if result < 0.5 else 1``
        #print(result)
        frame = cv.putText(frame,
                        content, org=(x1, y1), 
                        fontFace=cv.FONT_HERSHEY_SIMPLEX, 
                        fontScale=1, 
                        color=color, 
                        thickness=2)
        frame  = cv.rectangle(frame, 
                            pt1=(x1, y1), 
                            pt2=(x2, y2), 
                            color=color, 
                            thickness=2)

    cv.imshow('camera', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
vid.release()
cv.destroyAllWindows()
