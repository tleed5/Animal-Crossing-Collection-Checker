#!/usr/bin/env python3
"""Testy test
"""
# -*- encoding: utf-8 -*-
import cv2
import numpy as np

img_rgb = cv2.imread('images/bug-1.jpg')
template = cv2.imread('templates/bug.png', 0)
h, w = template.shape[::-1]

img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

font = cv2.FONT_HERSHEY_SIMPLEX

res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.75
loc = np.where( res >= threshold)
count = 0
mask = np.zeros(img_gray.shape[:2], np.uint8)
for pt in zip(*loc[::-1]):
    if mask[pt[1] + h//2, pt[0] + w//2] != 255:
        mask[pt[1]:pt[1]+h, pt[0]:pt[0]+w] = 255
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,255,0), 1)
        cv2.putText(img_rgb, str(count), (pt[0], pt[1]), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        count += 1
print(mask)
cv2.imshow('Color', img_rgb)
cv2.imshow("Threshold", img_gray)
cv2.imshow("Template", template)
cv2.waitKey(0)
cv2.destroyAllWindows()