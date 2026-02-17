import cv2
import card_detection as cd
import classification as cl
import time

user_input = input("Do you want to show your cards? (y/n): ").lower()

if user_input == "y":
    cap = cv2.VideoCapture(0)
    card_detection = cd.CardDetector()

    current_time = time.time()

    while True:
        ret, frame = cap.read()

        try:
            # Use Method 1
            if time.time() - current_time > 10:
                raise TimeoutError
            cv2.putText(frame, "Trying the first method", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            cv2.putText(frame, str(10 - int(time.time() - current_time)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
            detected_card = card_detection.detect(frame)

        except TimeoutError:
            # Use Method 2
            cv2.putText(frame, "Place the card in the rectangle", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            detected_card = card_detection.frame_crop_detect(frame)

        cv2.imshow("Card Detector", frame)

        card_classifier = cl.CardClassifier()
        detected_card_name = None
        if detected_card is not None:
            detected_card_name = card_classifier.template_match(detected_card)

        if detected_card is not None and detected_card_name is not None:
            cv2.imshow("Detected Card", detected_card)
            cv2.putText(detected_card, detected_card_name, (detected_card.shape[0] / 2, 0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()