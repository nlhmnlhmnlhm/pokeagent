import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import logging

# Set up a logger for the verifier
log = logging.getLogger(__name__)

class ScreenshotVerifier:
    """
    Compares two screenshots to verify if an action caused a
    significant and expected change using SSIM.
    """
    def __init__(self, change_threshold=0.95, no_change_threshold=0.99):
        """
        Initializes the verifier.
        
        Args:
            change_threshold (float): An SSIM score *below* this value is 
                                      considered a successful change.
            no_change_threshold (float): An SSIM score *above* this value is 
                                         considered a successful "no change" (for 'WAIT').
        """
        self.change_threshold = change_threshold
        self.no_change_threshold = no_change_threshold
        log.info(f"ScreenshotVerifier initialized with change_threshold={change_threshold}, no_change_threshold={no_change_threshold}")

    def _preprocess_image(self, img: Image.Image) -> np.ndarray:
        """
        Converts a PIL image into a format suitable for SSIM comparison.
        - Converts to grayscale (removes color noise).
        - Resizes to a smaller, fixed size to speed up comparison and 
          blur minor, insignificant details.
        """
        if img is None:
            log.warning("Cannot preprocess a None image.")
            return None
            
        # Convert to grayscale ('L' mode) and resize to a small, fixed size.
        # Resizing makes the comparison faster and less sensitive to tiny noise.
        try:
            img_gray = img.convert("L").resize((120, 80), Image.Resampling.BILINEAR)
            return np.array(img_gray)
        except Exception as e:
            log.error(f"Error preprocessing image: {e}")
            return None

    def compare_screenshots(self, img_before: Image.Image, img_after: Image.Image) -> float:
        """
        Compares two PIL screenshots and returns their structural similarity score.

        Returns:
            float: A score between 0.0 and 1.0. 
                   1.0 means identical, < 1.0 means different.
                   Returns 1.0 (identical) if comparison fails.
        """
        # Preprocess both images
        before_arr = self._preprocess_image(img_before)
        after_arr = self._preprocess_image(img_after)

        if before_arr is None or after_arr is None:
            log.warning("Could not compare screenshots, one or both images were invalid.")
            return 1.0  # Treat as identical if we can't process

        # Calculate SSIM
        # data_range is the dynamic range of the pixel values (0-255 for 8-bit grayscale)
        try:
            score = ssim(before_arr, after_arr, data_range=255)
            return score
        except Exception as e:
            log.error(f"Error calculating SSIM: {e}")
            return 1.0 # Treat as identical on error

    def verify_action(self, img_before: Image.Image, img_after: Image.Image, action: str) -> bool:
        """
        Verifies if an action succeeded by comparing the screenshots.

        Args:
            img_before: The PIL Image *before* the action.
            img_after: The PIL Image *after* the action.
            action: The string name of the action (e.g., "A", "UP", "WAIT").

        Returns:
            True if the action is considered successful, False otherwise.
        """
        if action is None:
            action = "WAIT"
            
        action = action.upper()
        
        # 1. Calculate the similarity score
        similarity_score = self.compare_screenshots(img_before, img_after)
        log.debug(f"Action: {action}, SSIM Score: {similarity_score:.4f}")

        # 2. Apply logic based on the action
        
        if action == "WAIT":
            return True
        
        elif action in ["A", "B", "START", "SELECT", "UP", "DOWN", "LEFT", "RIGHT", "L", "R"]:
            # SUCCESS for any other button press means the screen *DID* change.
            # We check if the score is *below* the change_threshold.
            is_successful = (similarity_score < self.change_threshold)
            if not is_successful:
                # This is a common case (e.g., pressing "A" on no dialog, running into a wall)
                log.info(f"Action {action} did not cause a significant screen change (Score: {similarity_score:.4f})")
            return is_successful
        
        else:
            # For any unknown action, default to the "change" logic
            log.warning(f"Unknown action '{action}', defaulting to 'change' logic.")
            return similarity_score < self.change_threshold