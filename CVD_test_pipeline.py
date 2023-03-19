from CVD import CVD
from CVD_decoder import Decoder


if __name__ == "__main__":
    #rate = (4, 16, 16)
    rate = (1, 1, 1)

    cvd = CVD(32, 8, *rate)

    cvd.encode("./video/coko.webm")

    res = cvd.approx(p=0.95)
    #cvd.write_compressed("./video/compresz")

    decoder = Decoder()
    decoder.decode("./video/compresz.npz")

    output_name = f"compressd_.gif"
    decoder.write_gif(output_name)
