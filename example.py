# -*- coding: utf-8 -*-

from scanner_lib import *
import cv2

# Find all image files under ../img
pg = IMGS_DIR.glob('**/*')
img_paths = [x for x in pg if x.is_file()]

# Initialize result buffer
img_results =[]
for f in img_paths:
    # Step 1: open file
    try:
        img_in = cv2.imread(str(f)).copy()
    except:
        raise Exception("Image file not exist or not supported")

    # Step 2: create edge map
    img_edged = edge_img(img_in, gaus_size=3)
    
    # Step 3: estimate document contour
    img_poly = contour_img(img_in,img_edged)
    
    # Step 4: perspective transform image
    try:
        img_warped = four_point_warp(img_in, img_poly)
    except:
        img_results.append(img_in)
        print("Failed to find correct object contour, drawing "
              "contour instead. File:", str(f))
        continue
    
    # Step 5: post process image
    img_pretty = beatify_scan(img_warped)
    
    # Step 6: fill in result buffer
    img_results.append(img_pretty)

# To use matplotlib show results
show_img(*img_results)

# To output results to files
# for i, r in enumerate(img_results):
#     rpath = RESULTS_DIR.joinpath(img_paths[i].name)
#     cv2.imwrite(str(rpath), r)