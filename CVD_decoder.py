import cv2
import imageio
from functions_CVD import *

class Decoder:

    def __init__(self, block_size=0, rY=1, rCr=1, rCb=1):
        self.block_size = block_size
        self.rY = rY
        self.rCr = rCr
        self.rCb = rCb
        self.decoded_video = None
        self.frame_rows = None
        self.frame_cols = None
        self.fps = None

        self.SVD = None

        self.n_blocks_vertical = None
        self.n_blocks_horizontal = None

        self.subblock_size = None


    def decode(self, path):

        print("decoding ...")

        SVD_load = np.load("./video/compresz.npz")
        dtp = np.float64

        US = SVD_load['US'].astype(dtp) #+ np.finfo(dtp).eps
        VT = SVD_load['VT'].astype(dtp) #+ np.finfo(dtp).eps

        datatype = np.int64
        n = VT.shape[1]
        valid_range = np.iinfo(datatype).max * 0.8
        VT = (VT / valid_range) / np.sqrt(n)

        matrix = US @ VT

        # debug
        matrix_load = np.load("./videomatblocks.npz")
        matrix = matrix_load['vm']


        params = SVD_load['params']
        self.fps = params[0]
        (self.block_size, self.rY, self.rCr, self.rCb, self.frame_rows, self.frame_cols, self.n_blocks_horizontal,
         self.n_blocks_vertical, self.subblock_size) = params[1:].astype(int)


        decoded = []  # TODO prealkoacia
        if self.block_size == 0:
            for frame in matrix.T:
                # TODO zjednotit
                if self.subblock_size == 0:
                    y, cr, cb = np.array_split(frame, 3)
                    y = np.around(y * self.rY)
                    cr = np.around(cr * self.rCr)
                    cb = np.around(cb * self.rCb)
                    # print(y)
                    YCrCb_im = np.dstack([vec.reshape(self.frame_rows, self.frame_cols) for vec in (y, cr, cb)])
                else:
                    YCrCb_im = reconst_subsampled_block(frame, self.subblock_size)

                # TODO rovnomerna transformacia do 8bit
                YCrCb_im[YCrCb_im <= 0] = 0
                YCrCb_im[YCrCb_im >= 255] = 255
                YCrCb_im = (np.rint(YCrCb_im)).astype(np.uint8)
                # cv2.imshow('FrameYCC', YCrCb_im)
                # cv2.waitKey(0)
                bgr_im = cv2.cvtColor(YCrCb_im, cv2.COLOR_YCrCb2BGR)
                decoded.append(bgr_im)

        else:
            blocks_per_frame = self.n_blocks_vertical * self.n_blocks_horizontal
            print("blocks per frame", blocks_per_frame, matrix.shape[1])
            first_block = 0
            for last_block in range(blocks_per_frame, matrix.shape[1], blocks_per_frame):
                blocks = matrix[:, first_block:last_block]
                frame = np.empty((0, self.block_size*self.n_blocks_horizontal, 3))

                block_i = 0
                for j in range(self.n_blocks_vertical):
                    blockrow = np.empty((self.block_size, 0, 3))
                    for i in range(self.n_blocks_horizontal):

                        block = blocks[:, block_i]
                        block_i += 1

                        if self.subblock_size == 0:
                            y, cr, cb = np.array_split(block, 3)
                            y = np.around(y * self.rY)
                            cr = np.around(cr * self.rCr)
                            cb = np.around(cb * self.rCb)
                            # print(y)
                            YCrCb_block = np.dstack([vec.reshape(self.frame_rows, self.frame_cols) for vec in (y, cr, cb)])
                        else:
                            YCrCb_block = reconst_subsampled_block(block, self.block_size)

                        # TODO rovnomerna transformacia do 8bit
                        YCrCb_block[YCrCb_block <= 0] = 0
                        YCrCb_block[YCrCb_block >= 255] = 255
                        #cv2.imshow('FrameYCC', YCrCb_block)
                        #cv2.waitKey(0)

                        blockrow = np.hstack((blockrow, YCrCb_block))

                    frame = np.vstack((frame, blockrow))

                frame = frame[:self.frame_rows, :self.frame_cols, :]
                frame = frame.astype(np.uint8)
                rgbframe = cv2.cvtColor(frame, cv2.COLOR_YCR_CB2BGR)
                decoded.append(rgbframe)
                first_block = last_block

        self.decoded_video = decoded
        print("decoded")

    def write_gif(self, filename):
        print(f"saving to {filename}")

        with imageio.get_writer(filename, mode='I') as writer:
            frames = self.decoded_video
            for fr_bgr in frames:
                frame = cv2.cvtColor(fr_bgr, cv2.COLOR_BGR2RGB)
                writer.append_data(frame)

        print("file saved")



if __name__ == "__main__":
    decoder = Decoder()
    decoder.decode("./video/compresz.npz")

    output_name = f"compressd_.gif"
    decoder.write_gif(output_name)
