import cv2
import numpy as np
import json
import os
import time

def sort_points_clockwise(keyps):
    sorted_index = [0, 1, 2, 3]
    start_p = keyps[0]
    vec_1 = keyps[1] - start_p
    vec_2 = keyps[2] - start_p
    vec_3 = keyps[3] - start_p
    angle_1_2 = np.cross(vec_1, vec_2)
    angle_1_3 = np.cross(vec_1, vec_3)
    angle_2_3 = np.cross(vec_2, vec_3)
    if angle_1_2 < 0:
        sorted_index[1], sorted_index[2] = sorted_index[2], sorted_index[1]
    if angle_1_3 < 0:
        sorted_index[1], sorted_index[3] = sorted_index[3], sorted_index[1]
    if angle_2_3 < 0:
        sorted_index[2], sorted_index[3] = sorted_index[3], sorted_index[2]
    return keyps[sorted_index]

def project_rectify(img, poly : np.ndarray, scale=1):
    poly = sort_points_clockwise(poly)
    poly = poly.astype(np.float32)
    oriented_rec = cv2.minAreaRect(poly)
    oriented_points = cv2.boxPoints(oriented_rec)
    min_dis = 1e10
    base_ = np.array([0, 1, 2, 3])
    align_poly = 0
    for cnt in range(4):
        sum_dis = np.linalg.norm(oriented_points - poly[base_], axis=1).sum()
        if sum_dis < min_dis:
            align_poly = poly[base_]
            min_dis = sum_dis
        base_ = np.roll(base_, 1)
    w, h = max(oriented_rec[1]), min(oriented_rec[1])
    if w > 1.5*h:
        rec_w = int(w*scale)
        rec_h = int(w*2*scale/3)
    else:
        rec_w = int(3*h*scale/2)
        rec_h = int(h*scale)
    rec_sorted_points = np.array([
                                [0, 0],
                                [rec_w-1, 0],
                                [rec_w-1, rec_h-1],
                                [0, rec_h-1]
                                ], np.float32)
    if oriented_rec[1][0] > oriented_rec[1][1]:
        align_poly = align_poly[[1, 2, 3, 0]]
    else:
        align_poly = align_poly[[2, 3, 0, 1]]
    warpR = cv2.getPerspectiveTransform(align_poly, rec_sorted_points)
    roi_img = cv2.warpPerspective(img, warpR, (rec_w, rec_h),borderMode=cv2.BORDER_REPLICATE)
    return roi_img

if __name__ == "__main__":
    # sds_path = "data/train_val_data/val/aug_val.sds"
    # with open(sds_path, "r") as f:
    #     lines = f.readlines()
    # for line in lines:
    #     json_line = json.loads(line.strip())
    #     file_name = json_line["file_name"]
    #     keypoints = json_line["annotations"]
    #     img = cv2.imread(os.path.join("data/train_val_data/val/aug_val_imgs",file_name))
    #     for keyp in keypoints:
    #         keypoint = np.array(keyp["keypoint"])
    #         begin_time = time.time()
    #         roi = project_rectify(img, keypoint)
    #         print("latency is: ", time.time()-begin_time)
    #         cv2.imshow("", roi)
    #         print(roi.shape)
    #         if cv2.waitKey(0) == ord("q"):
    #             exit()
    keypoint= np.array([
        [
          274.57142857142856,
          903.5714285714284
        ],
        [
          2424.5714285714284,
          1253.5714285714284
        ],
        [
          2863.8571428571427,
          2435.7142857142853
        ],
        [
          238.85714285714278,
          2424.9999999999995
        ]
      ])
    img = cv2.imread("/Users/lili/Downloads/IMG_6851.JPG")
    roi = project_rectify(img, keypoint) 
    cv2.imshow("", roi)   
    if cv2.waitKey(0) == ord("q"):
        exit()
