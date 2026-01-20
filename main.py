import cv2
import card_detection as cd

user_input = input("Do you want to show your cards? (y/n): ").lower()

if user_input == "y":
    cap = cv2.VideoCapture(0)
    card_detection = cd.CardDetector()
    while True:
        ret, frame = cap.read()
        detected_card = card_detection.detect(frame)
        cv2.imshow("Card Detector", frame)
        if detected_card is not None:
            cv2.imshow("Detected Card", detected_card)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()