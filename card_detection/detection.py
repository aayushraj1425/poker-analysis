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
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        return thresh


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
        src_pts = np.float32(approx.reshape(4, 2))
        dst_pts = np.float32([[0, 0], [CARD_WIDTH, 0], [CARD_WIDTH, CARD_HEIGHT], [0, CARD_HEIGHT]])

        m = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped_img = cv2.warpPerspective(frame, m, (CARD_WIDTH, CARD_HEIGHT))
        return warped_img

    def detect(self, frame):
        thresh = self.pre_image_process(frame)
        card_approx = self.find_contours(frame, thresh)
        if card_approx is not None:
            detected_img = self.picture_transform(frame, card_approx)
            return detected_img
        return None