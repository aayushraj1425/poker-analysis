import cv2
import numpy as np

CARD_WIDTH = 120
CARD_HEIGHT = 175

class CardDetector:
    """
    Detects playing cards in a video frame.
    """

    @staticmethod
    def pre_image_process(frame):
        """
        Converts frame to a binary image.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_clahe = clahe.apply(gray)

        blur = cv2.GaussianBlur(gray_clahe, (5, 5), 0)

        _, thresh = cv2.threshold(blur, 225, 255, cv2.THRESH_BINARY)

        v_low, v_high = np.percentile(blur, [25, 75])
        lower = int(v_low)
        upper = int(v_high)
        edges = cv2.Canny(blur, lower, upper)

        kernel = np.ones((3, 3), np.uint8)
        edges_dilated = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
        edges_closed = cv2.morphologyEx(edges_dilated, cv2.MORPH_CLOSE, kernel)

        combined = cv2.bitwise_or(thresh, edges_closed)

        return combined

    @staticmethod
    def find_contours(frame, thresh):
        """
        Detects contours in the thresholded image and identifies playing cards.
        """
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        frame_area = frame.shape[0] * frame.shape[1]

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < 0.01 * frame_area:
                continue

            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)

            if len(approx) != 4:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            ratio = h / w if h > w else w / h

            if 1.3 <= ratio <= 1.5:
                cv2.drawContours(frame, [approx], -1, (0, 255, 0))
                x = approx.ravel()[0]
                y = approx.ravel()[1] - 10
                cv2.putText(frame, "Card Detected", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
                return approx
        return None

    @staticmethod
    def draw_rectangle(frame, start_point, end_point):
        """
        Draws rectangle in the center of the frame.
        """

        color = (255, 0, 0)
        thickness = 2

        cv2.rectangle(frame, start_point, end_point, color, thickness)

    def frame_crop_detect(self, frame):
        """
        Crops the image inside the rectangle and detects card.
        """
        h, w = frame.shape[:2]
        start_point = (int(w / 2 - CARD_WIDTH / 2), int(h / 2 - CARD_HEIGHT / 2))
        end_point = (int(start_point[0] + CARD_WIDTH), int(start_point[1] + CARD_HEIGHT))
        self.draw_rectangle(frame, start_point, end_point)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        card_approx = self.find_contours(frame, thresh)

        if card_approx is not None:
            detected_img = self.picture_transform(frame, card_approx)
            return detected_img
        return None

    @staticmethod
    def picture_transform(frame, approx):
        """
        Transforms the Detected image into a separate one.
        """
        src_pts = np.float32(approx.reshape(4, 2))
        dst_pts = np.float32([[0, 0], [CARD_WIDTH, 0], [CARD_WIDTH, CARD_HEIGHT], [0, CARD_HEIGHT]])

        m = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped_card = cv2.warpPerspective(frame, m, (CARD_WIDTH, CARD_HEIGHT))
        return warped_card

    def detect(self, frame):
        """
        Detects the card in the image.
        """
        thresh = self.pre_image_process(frame)
        card_approx = self.find_contours(frame, thresh)
        cv2.imshow("Thresh", thresh)
        if card_approx is not None:
            detected_img = self.picture_transform(frame, card_approx)
            return detected_img
        return None