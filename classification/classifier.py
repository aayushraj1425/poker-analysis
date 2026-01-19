import cv2 as cv
import numpy as np

class CardClassifier:
    """
    Classifies a detected playing card into rank and suit.
    """

    def extract_corner(self, warped_card):
        """
        Extracts the top-left corner of the card with rank and suit.
        """
        h, w = warped_card.shape[:2]
        corner = warped_card[
            int(0.02 * h):int(0.25 * h),
            int(0.02 * w):int(0.25 * w)
        ]

        return corner

    def preprocess_corner(self, corner):
        """
        Preprocess corner for classification.
        """
        gray = cv.cvtColor(corner, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray, (3,3), 0)
        _, thresh = cv.threshold(blur, 150, 255, cv.THRESH_BINARY_INV)
        return thresh

    def classify(self, warped_card):
        """
        Classify the given card.l
        """
        corner = self.extract_corner(warped_card)
        processed_corner = self.preprocess_corner(corner)

        rank = self.classify_rank(processed_corner)
        suit = self.classify_suit(processed_corner)