import cv2
import card_detection as cd

user_input = input("Do you want to show your cards? (y/n): ").lower()

if user_input == "y":
    cap = cv2.VideoCapture(0)
    card_detection = cd.CardDetector()
    while True:
        ret, frame = cap.read()
        card_detection.detect(frame)
        cv2.imshow("Card Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



