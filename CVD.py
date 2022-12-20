import numpy as np
import cv2
from sklearn.utils.extmath import randomized_svd as rSVD


class CVD:

    def __init__(self, rY=1, rCr=1, rCb=1):
        self.rY = rY
        self.rCr = rCr
        self.rCb = rCb
        self.video_matrix = None
        self.approx_matrix = None
        self.decoded_video = None
        self.frame_rows = None
        self.frame_cols = None
        self.fps = None

        self.SVD = None


    def encode(self, path):
        cap = cv2.VideoCapture(path)
        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
        if int(major_ver) < 3:
            fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
        else:
            fps = cap.get(cv2.CAP_PROP_FPS)
        print(fps)
        self.fps = fps

        vid = []
        ret = True
        while cap.isOpened() and ret:
            ret, frame = cap.read()
            if ret:
                YCrCb_im = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
                y, cr, cb = cv2.split(YCrCb_im)
                y //= self.rY
                cr //= self.rCr
                cb //= self.rCb
                if self.frame_rows is None and self.frame_cols is None:
                    self.frame_rows, self.frame_cols = y.shape
                    print(self.frame_rows, self.frame_cols)
                col = np.concatenate((y.ravel(), cr.ravel(), cb.ravel()))
                vid.append(col)

        video_matrix = np.array(vid, order='c').T
        self.video_matrix = video_matrix
        print(self.video_matrix.shape)
        np.savez_compressed("videomoat", vm=video_matrix.astype(int))


    def approx(self, p=0.9, step=50):
        matrix = self.video_matrix.copy().astype(float)
        sigma_sum = 0               # sucet zahrnuteho spektra
        sigmas_pred = 1             # odhad na sucet celeho spektra
        n_rows, n_cols= matrix.shape
        bottleneck = (n_rows * n_cols) / (n_rows + n_cols + 1)

        n = min(n_rows, n_cols)
        k = 0

        result = np.zeros_like(matrix).astype(float)
        print(type(result))

        res_u, res_s, res_vt = None, None, None

        while (sigma_sum / sigmas_pred) < p:
            # print("accuracy pred:", sigma_sum / sigmas_pred)
            print('-', end='')
            U, S, VT = rSVD(matrix.copy(), step, random_state=None)
            k += step
            sigma_sum += sum(S)
            sigmas_pred = sigma_sum + min(S) * (n - k)
            approx = U @ (VT * S[:, None])
            result += approx
            matrix -= approx

            if res_u is None:
                res_u, res_s, res_vt = U, S, VT
            else:
                res_u = np.concatenate((res_u, U), axis=1)
                res_s = np.concatenate((res_s, S))
                res_vt = np.concatenate((res_vt, VT))

            if k > bottleneck:
                print(f"COMPRESSION INEFFICIENT at rank = {k}")


        print()
        print(k)
        print("result accuracy pred:", sigma_sum / sigmas_pred)
        self.SVD = res_u, res_s, res_vt
        return res_u, res_s, res_vt

    def write_compressed(self, filename):
        print("video shape", self.video_matrix.shape)
        u, s, vt = self.SVD
        s = np.ravel(s)
        print("compressed shape", u.shape, s.shape, vt.shape)
        np.savez_compressed(filename, U=u, S=s, VT=vt)

    def decode(self, matrix):
        print("decoding ...")
        decoded = []
        for col in matrix.T:
            y, cr, cb = np.array_split(col, 3)
            y = np.around(y*self.rY)
            cr = np.around(cr*self.rCr)
            cb = np.around(cb*self.rCb)
            YCrCb_im = np.dstack([vec.reshape(self.frame_rows, self.frame_cols) for vec in (y, cr, cb)])

            # TODO rovnomerna transformacia do 8bit
            YCrCb_im[YCrCb_im <= 0] = 0
            YCrCb_im[YCrCb_im >= 255] = 255
            YCrCb_im = (np.rint(YCrCb_im)).astype(np.uint8)
            #cv2.imshow('FrameYCC', YCrCb_im)
            #cv2.waitKey(0)
            bgr_im = cv2.cvtColor(YCrCb_im, cv2.COLOR_YCrCb2BGR)
            decoded.append(bgr_im)
            #cv2.imshow('Frame', bgr_im)
            #cv2.waitKey(0)
        self.decoded_video = decoded
        print("decoded")

    def write_mp4(self, filename):
        print(f"saving to {filename}")
        out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), round(self.fps), (self.frame_cols, self.frame_rows))
        frames = self.decoded_video
        for frame in frames:
            out.write(frame)
        out.release()
        print("file saved")


if __name__ == "__main__":
    rate = (2, 4, 4)
    rank = 50
    #output_name = f"compressd_rate-{''.join(str(num) for num in rate)}_rank-{rank}.mp4"
    cvd = CVD(*rate)
    cvd.encode("coko.webm")
    res = cvd.approx(p=0.5)
    cvd.write_compressed("compresz")
    #cvd.decode(cvd.approx_matrix)
    #cvd.write_mp4(output_name)



