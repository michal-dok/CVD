import numpy as np
import numpy.linalg as LA
import cv2
from sklearn.utils.extmath import randomized_svd as rSVD


class CVD:

    def __init__(self, ):
        self.video_matrix = None
        self.frame_rows = None
        self.frame_cols = None
        self.fps = None
        self.compressed = None

    def encode(self, path, rate=(1, 1, 1)):
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
                ycbcr_im = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
                y, cb, cr = cv2.split(ycbcr_im)
                y //= rate[0]
                cb //= rate[1]
                cr //= rate[2]
                if self.frame_rows is None and self.frame_cols is None:
                    self.frame_rows, self.frame_cols = y.shape
                    print(self.frame_rows, self.frame_cols)
                col = np.concatenate((y.ravel(), cb.ravel(), cr.ravel()))
                vid.append(col)

        video_matrix = np.array(vid, order='c').T
        self.video_matrix = video_matrix
        print(self.video_matrix.shape)


    def compress(self, rank=50):
        print("compressing ...")
        U, S, VT = LA.svd(self.video_matrix, False)
        print("SVD done!")

        k = rank
        Ar = np.dot(U[:, :k], np.dot(np.diag(S[:k]), VT[:k, :]))

        #Ar = np.zeros((len(U), len(VT)))
        #for i in range(rank):
        #    Ar += S[i] * np.outer(U[i], VT[i])

        self.video_matrix = Ar
        print("compression done!")


    def decode(self, rate=(1, 1, 1)):
        print("decoding ...")
        decoded = []
        for col in self.video_matrix.T:
            y, cr, cb = np.array_split(col, 3)
            y = np.around(y*rate[0])
            cr = np.around(cr*rate[1])
            cb = np.around(cb*rate[2])
            ycbcr_im = np.dstack([vec.reshape(self.frame_rows, self.frame_cols) for vec in (y, cr, cb)])
            ycbcr_im[ycbcr_im <= 0] = 0
            ycbcr_im[ycbcr_im >= 255] = 255
            ycbcr_im = (np.rint(ycbcr_im)).astype(np.uint8)

            print("shape:", ycbcr_im.shape)
            print(np.max(ycbcr_im), np.min(ycbcr_im))
            #cv2.imshow('FrameYCC', ycbcr_im)
            #cv2.waitKey(0)
            bgr_im = cv2.cvtColor(ycbcr_im, cv2.COLOR_YCrCb2BGR)
            decoded.append(bgr_im)
            #cv2.imshow('Frame', bgr_im)
            #cv2.waitKey(0)
        self.compressed = decoded
        print("decoded")

    def write(self, filename):
        print(f"saving to {filename}")
        out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), round(self.fps), (self.frame_cols, self.frame_rows))
        frames = self.compressed
        for frame in frames:
            out.write(frame)
        out.release()
        print("file saved")


if __name__ == "__main__":
    rate = (2, 4, 4)
    rank = 50
    output_name = f"compressd_rate-{''.join(str(num) for num in rate)}_rank-{rank}.mp4"
    cvd = CVD()
    cvd.encode("coko.webm", rate=rate)
    cvd.compress(rank=rank)
    cvd.decode(rate=rate)
    cvd.write(output_name)



