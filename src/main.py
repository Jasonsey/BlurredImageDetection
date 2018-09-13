from pathlib import Path
from api.cv2.detection import detection


input_path = Path('../data/input/waimailicense')
output_path = Path('../data/output')
detection(input_path, output_path)
