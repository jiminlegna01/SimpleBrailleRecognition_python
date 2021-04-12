import sys
import math
import cv2
import numpy as np
import time


class Braille:
    count = 0

    def __init__(self, rect, value):
        self.rect = rect        # [point of top-left, point of bottom-right]
        self.value = value      # 6bit binary(order : top-left -> bottom-left -> top-right -> bottom-right)
        Braille.count += 1


def main(argv):
    # 0) define param
    margin = 10
    color_black = np.zeros(3, dtype=np.uint8)

    # 1) read input image
    start = time.time()
    input_img = cv2.imread('braille_sample1.png', cv2.IMREAD_GRAYSCALE)
    end = time.time()
    print("1) read input image :", round(((end - start) * 1000), 2), "ms")
    cv2.imshow('(1) input_img', input_img)

    # 2) pre-processing input image
    # apply adaptive threshold
    start = time.time()
    threshold_img = cv2.adaptiveThreshold(input_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10)
    # apply gaussian blur & erode and then threshold again
    kernel = np.ones((3, 3), dtype=np.uint8)  # cpp style : kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    threshold_img = cv2.GaussianBlur(threshold_img, (3, 3), 0)
    threshold_img = cv2.erode(threshold_img, kernel)
    ret, threshold_img = cv2.threshold(threshold_img, 21, 255, cv2.THRESH_BINARY)
    cv2.imshow('(1.5) threshold_img', threshold_img)
    # make margin and resize from original image
    v, h = threshold_img.shape
    measure_img = np.ones((v + margin * 2, h + margin * 2), dtype=np.uint8) * 255
    measure_img[margin:v + margin, margin:h + margin] = threshold_img.copy()
    measure_img = cv2.resize(measure_img, (measure_img.shape[1] * 2, measure_img.shape[0] * 2))
    end = time.time()
    print("2) pre-processing input image :", round(((end - start) * 1000), 2), "ms")
    cv2.imshow('(2) measure_img', measure_img)

    # 3) detect blobs
    start = time.time()
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 2.0 * 2.0
    params.maxArea = 20.0 * 20.0
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(measure_img)
    detected_img = cv2.drawKeypoints(measure_img, keypoints, np.array([]), (0, 0, 255),
                                     cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    end = time.time()
    print("3) detect blobs :", round(((end - start) * 1000), 2), "ms")
    cv2.imshow('(3) detected_img', detected_img)

    # 4) normalize keypoints to coordinate line set
    start = time.time()
    blob_size_sum = 0.0
    for k in keypoints:
        blob_size_sum += k.size
    blob_size = blob_size_sum / len(keypoints)
    # print("average of blob size : ", blob_size)
    coord_x = np.array([], dtype=np.uint16)
    coord_y = np.array([], dtype=np.uint16)
    for k in keypoints:
        is_new = True
        for x in coord_x:
            if abs(x - k.pt[0]) < blob_size:
                is_new = False
        if is_new is True:
            coord_x = np.append(coord_x, round(k.pt[0]))
        is_new = True
        for y in coord_y:
            if abs(y - k.pt[1]) < blob_size:
                is_new = False
        if is_new is True:
            coord_y = np.append(coord_y, round(k.pt[1]))
    coord_x.sort()
    coord_y.sort()
    coordinate_img = detected_img.copy()
    for x in coord_x:
        cv2.line(coordinate_img, (x, 0), (x, coordinate_img.shape[0]), (255, 0, 0))
    for y in coord_y:
        cv2.line(coordinate_img, (0, y), (coordinate_img.shape[1], y), (255, 0, 0))
    end = time.time()
    print("4) normalize keypoints to coordinate line set :", round(((end - start) * 1000), 2), "ms")
    cv2.imshow('(4) coordinate_img', coordinate_img)

    # 5) move keypoints to the nearest coordinate point
    start = time.time()
    edited_key_array = np.array([], dtype=np.uint16)
    for k in keypoints:
        distance_x = detected_img.shape[1] / 2
        distance_y = detected_img.shape[0] / 2
        temp_x = 0
        temp_y = 0
        for x in coord_x:
            if distance_x > abs(k.pt[0] - x):
                distance_x = abs(k.pt[0] - x)
                temp_x = x
        for y in coord_y:
            if distance_y > abs(k.pt[1] - y):
                distance_y = abs(k.pt[1] - y)
                temp_y = y
        # same : after all append, and then edited_key_array = edited_key_array.reshape(-1, 2)
        if edited_key_array.size == 0:
            edited_key_array = np.append(edited_key_array, [temp_x, temp_y])
        else:
            edited_key_array = np.vstack([edited_key_array, [temp_x, temp_y]])
    # make image from the edited keypoint set
    edit_img = np.ones(detected_img.shape, np.uint8) * 255
    circle_size = round(blob_size / 3)
    for k in edited_key_array:
        cv2.circle(edit_img, tuple(k), circle_size, (0, 0, 0), -1, cv2.LINE_AA)
    end = time.time()
    print("5) move keypoints to the nearest coordinate point :", round(((end - start) * 1000), 2), "ms")
    cv2.imshow('(5) edit_img', edit_img)

    # 6) segmentation braille rectangle
    start = time.time()
    braille_array = np.array([])
    segmentation_img = edit_img.copy()
    # ignore to consider the start position of braille
    # start_pos = 0
    # if (coord_x[1] - coord_x[0]) > (coord_x[2] - coord_x[1]):
    #     start_pos = 1
    rect_margin = int(math.ceil(blob_size / 2))
    rect_width = 0
    for i in range(0, coord_y.size, 3):
        for j in range(0, coord_x.size, 2):
            rect = [(coord_x[j] - rect_margin, coord_y[i] - rect_margin),
                    (coord_x[j + 1] + rect_margin, coord_y[i + 2] + rect_margin)]
            rect_width += rect[1][0] - rect[0][0]
            cv2.rectangle(segmentation_img, rect[0], rect[1], (0, 0, 255))
            value = np.zeros(6, dtype=np.uint8)
            value_index = 0
            for k in range(2):
                for m in range(3):
                    if (edit_img[coord_y[i + m], coord_x[j + k]] == color_black).all():
                        value[value_index] = 1
                    value_index += 1
            braille_array = np.append(braille_array, Braille(rect, value))
    rect_width /= Braille.count
    end = time.time()
    print("6) segmentation braille rectangle :", round(((end - start) * 1000), 2), "ms")
    cv2.imshow('(6) segmentation_img', segmentation_img)
    # print("total braille count : ", Braille.count)
    # for b in braille_array:
    #     print("rect : ", b.rect, " value : ", b.value)

    '''
        # comparision with the input image
        compare_img = cv2.cvtColor(input_img, cv2.COLOR_GRAY2BGR)
        width = int(segmentation_img.shape[1] / 2)
        height = int(segmentation_img.shape[0] / 2)
        print(width, height)
        resized_img = segmentation_img.copy()
        resized_img = cv2.resize(resized_img, (width, height))
        print(resized_img.shape, compare_img.shape, resized_img[margin:height - margin, margin:width - margin].shape)
        cv2.addWeighted(compare_img, 0.8, resized_img[margin:height - margin, margin:width - margin], 0.2, 0.0, compare_img)
        cv2.imshow("(6.5)compare_img", compare_img)
    '''

    # 7) make result image
    start = time.time()
    result_img = np.ones(segmentation_img.shape, dtype=np.uint8) * 255
    cv2.addWeighted(result_img, 0.8, segmentation_img, 0.2, 0.0, result_img)
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = rect_width / 60.0
    font_thickness = round(font_scale * 2)
    for b in braille_array:
        tl = np.array(b.rect[0])
        br = np.array(b.rect[1])
        center = np.mean(np.array([tl, br]), axis=0, dtype=np.uint16)
        bl = np.array([b.rect[0][0], b.rect[1][1] + round(font_scale * 20)])
        value_str = ''
        int_str = 0
        for t in b.value:
            value_str += str(t)
            int_str = int(value_str, 2)
        center[0] -= round(cv2.getTextSize(str(int_str), font_face, font_scale, font_thickness)[0][0] / 2)
        center[1] += round(cv2.getTextSize(str(int_str), font_face, font_scale, font_thickness)[0][1] / 2)
        cv2.putText(result_img, value_str, tuple(bl), font_face, font_scale / 2, (0, 0, 0), font_thickness)
        cv2.putText(result_img, str(int_str), tuple(center), font_face, font_scale, (255, 0, 0), font_thickness)
    end = time.time()
    print("7) make result image :", round(((end - start) * 1000), 2), "ms")
    cv2.imshow('(7) result_img', result_img)

    cv2.waitKey(0)


if __name__ == "__main__":
    main(sys.argv[1:])
