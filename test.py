import cv2
import dlib
import numpy as np
def recognize_face(face_descriptor, face_descriptors, threshold=0.6):
    # Tính toán khoảng cách Euclidean giữa face_descriptor và face_descriptors đã biết
    distances = np.linalg.norm(face_descriptors - face_descriptor, axis=1)
    
    # Tìm chỉ mục của face_descriptor có khoảng cách nhỏ nhất
    min_distance_idx = np.argmin(distances)
    
    # Kiểm tra xem khoảng cách nhỏ nhất có nhỏ hơn ngưỡng (threshold) hay không
    if distances[min_distance_idx] < threshold:
        return min_distance_idx
    else:
        return None
# Khởi tạo bộ nhận diện khuôn mặt dlib
detector = dlib.get_frontal_face_detector()

# Khởi tạo trình nhận dạng khuôn mặt dlib
shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_recognizer = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

# Đọc video từ camera máy
cap = cv2.VideoCapture(0)

while True:
    # Đọc từng frame từ video
    ret, frame = cap.read()
    
    # Chuyển đổi frame sang định dạng RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Phát hiện khuôn mặt trong frame
    face_rects = detector(rgb_frame)
    
    # Với mỗi khuôn mặt được phát hiện, nhận dạng và vẽ hình chữ nhật xung quanh khuôn mặt
    for rect in face_rects:
        landmarks = shape_predictor(rgb_frame, rect)
        face_descriptor = np.array(face_recognizer.compute_face_descriptor(rgb_frame, landmarks))
        
        # Gọi hàm nhận dạng khuôn mặt và in ra kết quả
        recognized_index = recognize_face(face_descriptor, known_face_descriptors)
        if recognized_index is not None:
            print("Khuôn mặt được nhận dạng!")
        else:
            print("Khuôn mặt không được nhận dạng!")
            
        # Vẽ hình chữ nhật xung quanh khuôn mặt
        (x, y, w, h) = (rect.left(), rect.top(), rect.width(), rect.height())
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    # Hiển thị frame kết quả
    cv2.imshow('Face Recognition', frame)
    
    # Kiểm tra phím nhấn từ người dùng để thoát khỏi vòng lặp
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
