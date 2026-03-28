conf = 0.4          # minimum OCR confidence required to keep detected text (filters low-quality predictions)
tile_h = 800
tile_w = 800
tile_overlap = 30   # amount of pixels that tiles will overlap over
scale = 2.0         # image upscaling factor before detection to improve small text recognition
max_workers = 8     # number of parallel threads used for tile processing and OCR
box_condense = (20,4)   # thresholds (x,y) for merging nearby text boxes into a single box