import cv2
import numpy as np

class CardClassifier:
    """
    Classifies a detected playing card into rank and suit.
    """
    @staticmethod
    def extract_corner(warped_card):
        """
        Extracts the top-left corner of the card with rank and suit.
        """
        h, w = warped_card.shape[:2]
        corner = warped_card[
            int(0.02 * h):int(0.25 * h),
            int(0.02 * w):int(0.25 * w)
        ]

        return corner

    @staticmethod
    def preprocess_image(image):
        """
        Preprocess corner for classification.
        """
        color = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = color
        blur = cv2.GaussianBlur(gray, (3,3), 0)
        _, thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY_INV)
        return thresh

    def classify(self, warped_card):
        """
        Classify the detected card.
        """
        corner = self.extract_corner(warped_card)
        processed_corner = self.preprocess_image(corner)

        # rank = self.classify_rank(processed_corner)
        # suit = self.classify_suit(processed_corner)

    def template_match(self, warped_card):
        suits = ['H','D','C', 'S']
        ranks = ['A', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']

        corner = self.extract_corner(warped_card)
        processed_corner = self.preprocess_image(corner)

        card_templates = [rank + suit for suit in suits for rank in ranks]
        for card_template in card_templates:
            path_to_template = f"./templates/{card_template}.png"
            template = cv2.imread(path_to_template, cv2.IMREAD_GRAYSCALE)
            template = self.preprocess_image(template)

            w, h = template.shape[::-1]
            template = template[:int(0.25 * h), :(0.25 * w)]
            card = processed_corner.copy()

            res = cv2.matchTemplate(card, template, method=cv2.TM_CCOEFF_NORMED)
            match_threshold = 0.7
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if match_threshold <= max_val:
                top_left = max_loc
                bottom_right = (top_left[0] + w, top_left[1] + h)

                cv2.rectangle(card, top_left, bottom_right, 255,2)
                print(f"Matched card: {card_template}")