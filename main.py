import argparse
import os
import cv2
import time
import pickle
import cvzone
import numpy as np
import dlib
import datetime
from pygame import mixer
from imutils import face_utils
from scipy.spatial import distance
from yolo import YOLO

# Dlib yüz tespiti için kullanılan nesne
detectEye = dlib.get_frontal_face_detector()

# Dlib yüz şekli tahmin etme için kullanılan nesne
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Video kayıtları için kullanılan çıkış nesneleri
out_hand = None
out_drowsiness = None

def eye_aspect_ratio(eye):
    # Göz açıklık oranını hesaplamak için kullanılan fonksiyon
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Yüzdeki sol ve sağ gözleri belirlemek için kullanılan indeksler
lStart, lEnd = face_utils.FACIAL_LANDMARKS_68_IDXS['left_eye']
rStart, rEnd = face_utils.FACIAL_LANDMARKS_68_IDXS['right_eye']

class HandDetector:
    def __init__(self, network="normal", size=416, confidence=0.1, hands=4):
        # YOLO modeli ve diğer parametreleri başlatan sınıfın kurucu metodu
        self.network = network
        self.size = size
        self.confidence = confidence
        self.hands = hands
        self.width_warn_area, self.height_warn_area = 240, 330
        self.warning_area_color = (0, 255, 0)

        self.model_paths = {
            "normal": ("models/cross-hands.cfg", "models/cross-hands.weights"),
            "prn": ("models/cross-hands-tiny-prn.cfg", "models/cross-hands-tiny-prn.weights"),
            "v4-tiny": ("models/cross-hands-yolov4-tiny.cfg", "models/cross-hands-yolov4-tiny.weights"),
            "tiny": ("models/cross-hands-tiny.cfg", "models/cross-hands-tiny.weights")
        }

        # YOLO modelini yükleyen ve kamerayı başlatan bölüm
        if self.network not in self.model_paths:
            print(f"Invalid network type: {self.network}")
            exit()

        print(f"Loading {self.network}...")
        self.yolo = YOLO(*self.model_paths[self.network], ["hand"])

        # Kamera penceresini oluşturan bölüm
        print("Starting Video")
        cv2.namedWindow("Cam")
        self.cam = cv2.VideoCapture(0)

        # Uyarı alanı pozisyonlarını içeren dosyayı yükleme
        with open('WarnAreaPos', 'rb') as f:
            self.pos_list = pickle.load(f)

        self.width_warn_area, self.height_warn_area = 240, 330

        # Video kayıtları için çıkış nesnelerini başlatma
        self.out_hand = None
        self.out_drowsiness = None

    def draw_interface(self, frame, fps):
        # Arayüzü çizen fonksiyon
        cv2.rectangle(frame, (0, 0), (200, 70), (255, 255, 255), -1)
        cv2.putText(frame, f'FPS: {round(fps, 2)} ', (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(frame, f'Network: {self.network}', (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(frame, f'Confidence: {self.confidence}', (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    def check_warning_area(self, process, frame):
        # Uyarı alanındaki elleri kontrol eden fonksiyon
        for pos in self.pos_list:
            x, y = pos

            frame_crop = process[y: y + self.height_warn_area, x:x + self.width_warn_area]
            cv2.imshow(str(x * y), frame_crop)

            count = cv2.countNonZero(frame_crop)
            cvzone.putTextRect(frame, str(count), (x + 5, y + 20), scale=1.5, thickness=2, offset=0, colorR=(0, 0, 255))

            if count < 1000:
                colorWarn = (0, 255, 0)
            else:
                colorWarn = (0, 0, 255)
                cvzone.putTextRect(frame, str("DANGER"), (x + 145, y + 20), scale=1.5, thickness=2, offset=0,
                                   colorR=(0, 0, 255))

            thickness = 5
            cv2.rectangle(frame, pos, (pos[0] + self.width_warn_area, pos[1] + self.height_warn_area), colorWarn,
                          thickness)

    def run(self):
        # Uygulamayı çalıştıran fonksiyon
        mixer.init()
        mixer.music.load("music.wav")
        flag = 0
        thresh = 0.2
        frame_check = 10

        output_folder_drowsiness = 'DrowsinessDetections'
        output_folder_handing = 'HandDetections'
        os.makedirs(output_folder_drowsiness, exist_ok=True)
        os.makedirs(output_folder_handing, exist_ok=True)

        while True:
            if self.cam.get(cv2.CAP_PROP_POS_FRAMES) == self.cam.get(cv2.CAP_PROP_FRAME_COUNT):
                self.cam.set(cv2.CAP_PROP_POS_FRAMES, 0)

            is_opened, frame = self.cam.read()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_blur = cv2.GaussianBlur(frame_gray, (3, 3), 1)
            frame_threshold = cv2.adaptiveThreshold(frame_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                    cv2.THRESH_BINARY_INV, 25, 16)
            frame_median = cv2.medianBlur(frame_threshold, 5)
            kernel = np.ones((3, 3), np.int8)
            frame_dilate = cv2.dilate(frame_median, kernel, iterations=1)
            

            self.check_warning_area(frame_dilate, frame)

            start_time = time.time()

            # Kameradan bir çerçeve okuyamıyorsa hata mesajı ver ve döngüden çık
            if not is_opened:
                print("Error: Couldn't read a frame from the camera.")
                break

            # YOLO kullanarak nesne tespiti yapma
            width, height, time_scale, results = self.yolo.inference(frame)

            # Güvenilirlik eşiğine göre tespit sonuçlarını filtreleme
            confidence_threshold = float(self.confidence)
            results = [result for result in results if result[2] >= confidence_threshold]

            # Saniye başına çerçeve sayısını hesaplama
            fps = 1 / (time.time() - start_time)
            self.draw_interface(frame, fps)

            # Sonuçları güvenilirlik sırasına göre sıralama
            results.sort(key=lambda x: x[2])

            # Gösterilecek ellerin sayısını sınırlama
            hand_count = min(len(results), int(self.hands) if self.hands != -1 else len(results))

            subjects = detectEye(frame_gray, 0)
            for subject in subjects:
                shape = predict(frame_gray, subject)
                shape = face_utils.shape_to_np(shape)
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEar = eye_aspect_ratio(leftEye)
                rightEar = eye_aspect_ratio(rightEye)
                ear = (leftEar + rightEar) / 2.0
                """
                Here is the drawing eye contours
                
                
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1 ,(0,255,0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
                """
                if ear < thresh:
                    flag += 1
                    print(f"Eyes remained closed for {flag} seconds")
                    if self.out_drowsiness is None:
                        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
                        file_name = f'{output_folder_drowsiness}/drowsiness_detection_{current_time}.avi'
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        self.out_drowsiness = cv2.VideoWriter(file_name, fourcc, 5.0, (640, 480))

                    self.out_drowsiness.write(frame)

                    if flag >= frame_check:
                        cv2.putText(frame, "DROWSINESS DETECTED", (10, 460),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        mixer.music.play()
                else:
                    flag = 0

            for detect in results[:hand_count]:
                id, name, confidence, x, y, w, h = detect
                cx_hand = x + (w / 2)
                cy_hand = y + (h / 2)
                hand_in_warning_area = any(
                    x_warn <= cx_hand <= x_warn + self.width_warn_area and y_warn <= cy_hand <= y_warn + self.height_warn_area
                    for
                    x_warn, y_warn in self.pos_list)

                if hand_in_warning_area:
                    print(f"Hand in Warning Area! Coordinates: ({cx_hand}, {cy_hand})")
                    colorHand = (0, 255, 0)  # Set color to green

                    if self.out_hand is None:
                        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
                        file_name = f'{output_folder_handing}/hand_detection_{current_time}.avi'
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        self.out_hand = cv2.VideoWriter(file_name, fourcc, 5.0, (640, 480))

                    self.out_hand.write(frame)
                else:
                    cx = x + (w / 2)
                    cy = y + (h / 2)
                    print(f"Hand Coordinates: ({cx}, {cy})")
                    colorHand = (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x + w, y + h), colorHand, 2)
                text = "%s (%s)" % (name, round(confidence, 2))
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            frame_copy = frame.copy()
            cv2.imshow("Cam", frame_copy)

            key = cv2.waitKey(1)
            if key == 27:
                break

    def release_writers(self):
        # Video kayıt nesnelerini serbest bırakan fonksiyon
        if self.out_hand is not None:
            self.out_hand.release()

        if self.out_drowsiness is not None:
            self.out_drowsiness.release()

if __name__ == "__main__":
    # Komut satırı argümanlarını analiz etme
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--network', default="normal", choices=["normal", "tiny", "prn", "v4-tiny"])
    parser.add_argument('-s', '--size', default=416)
    parser.add_argument('-c', '--confidence', default=0.1)
    parser.add_argument('-nh', '--hands', default=4)
    args = parser.parse_args()

    # HandDetector sınıfını başlatma ve uygulamayı çalıştırma
    hand_detector = HandDetector(network=args.network, size=args.size, confidence=args.confidence, hands=args.hands)

    try:
        hand_detector.run()
    finally:
        hand_detector.release_writers()
