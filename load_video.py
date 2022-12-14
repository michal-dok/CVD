import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
import cv2
from sklearn.utils.extmath import randomized_svd as rSVD

print("f")
cap = cv2.VideoCapture("coko.webm")
print('s')
vid = []

ret = True
while cap.isOpened() and ret:
    ret, frame = cap.read()
    if ret:
        ycbcr_im = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
        y, cb, cr = cv2.split(ycbcr_im)
        col = np.concatenate((y.ravel(), cb.ravel(), cr.ravel()))
        vid.append(col)
        #cv2.imshow('Frame', ycbcr_im)
        #cv2.waitKey(0)

vid_arr = np.array(vid, order='c').T

print(vid_arr.shape)
print(y.shape)
print(rSVD(vid_arr, 5))
