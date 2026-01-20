import cv2
import numpy as np

CARD_WIDTH = 200
CARD_HEIGHT = 300


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

        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        _, thresh = cv2.threshold(blur, 225, 255, cv2.THRESH_BINARY)

        v = np.median(blur)
        lower = int(max(0, 0.66 * v))
        upper = int(min(255, 1.33 * v))
        edges = cv2.Canny(blur, lower, upper)

        kernel = np.ones((3, 3), np.uint8)
        edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

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
        thresh = self.pre_image_process(frame)
        card_approx = self.find_contours(frame, thresh)
        cv2.imshow("Thresh", thresh)
        if card_approx is not None:
            detected_img = self.picture_transform(frame, card_approx)
            return detected_img
        return None