import numpy as np
import cv2

def split_frame_to_blocks(frame, block_size):
    height, width, depth = frame.shape

    n_blocks_vertical = height // block_size + int(height % block_size > 0)
    n_blocks_horizontal = width // block_size + int(width % block_size > 0)

    missing_vertical = n_blocks_vertical * block_size - height
    missing_horizontal = n_blocks_horizontal * block_size - width


    if missing_horizontal:
        flipped_horiz = cv2.flip(frame[:, -missing_horizontal:, :], 1)
        stacked = np.hstack([frame, flipped_horiz])
    else:
        stacked = frame
    if missing_vertical:
        flipped_vert = cv2.flip(stacked[-missing_vertical:, :, :], 0)
        stacked = np.vstack([stacked, flipped_vert])

    tiles = [stacked[x:x + block_size, y:y + block_size, :] for x in range(0, stacked.shape[0], block_size) for y in
             range(0, stacked.shape[1], block_size)]

    return tiles


def spectrum_lm(sigmas):
    sigmas_log = np.log(sigmas.copy())

    k = len(sigmas)
    ln_range = np.log(np.arange(1, k + 1, 1))
    sum_ln_i = ln_range.sum()
    sum_ln_i2 = (ln_range ** 2).sum()
    X = np.array([np.ones(k), ln_range])

    det = k * sum_ln_i2 - sum_ln_i ** 2
    XTX_inv = [[sum_ln_i2, -sum_ln_i], [-sum_ln_i, k]] / det
    beta = XTX_inv @ (X @ sigmas_log)
    lnA, B_minus = beta

    return lnA, -B_minus


def get_rank_for_approx(initial_sigmas, full_rank, p=0.85):
    sigmas_included = initial_sigmas.sum()

    lnA, B = spectrum_lm(initial_sigmas)
    sigma_approx = lambda i: np.exp(lnA - B * np.log(i))
    k = len(initial_sigmas)

    sigmas_all = sigma_approx(np.arange(1, full_rank, 1))
    sigmas_all[:k] = initial_sigmas[:k]

    total_sum = sigmas_all.sum()

    approx_sum = 0
    for rank, sig in enumerate(sigmas_all):
        approx_sum += sig
        if (approx_sum / total_sum) >= p:
            break

    return rank, total_sum


def subsample_block(block, sub_size):
    luma, red, blue = cv2.split(block)
    start = sub_size // 2 -1
    Cr_sub = red[start::sub_size, start::sub_size]
    Cb_sub = blue[start::sub_size, start::sub_size]
    subsampled_vec = np.concatenate((luma.ravel(), Cr_sub.ravel(), Cb_sub.ravel()))
    return subsampled_vec


def reconst_subsampled_block(subsampled_vec, blocksize):
    y_vec = subsampled_vec[:blocksize**2]
    Cr_vec, Cb_vec = np.array_split(subsampled_vec[blocksize**2:], 2)
    blocksize_chroma = int(round(len(Cr_vec)**.5))
    y_block = y_vec.reshape(blocksize, blocksize)
    Cr_block = Cr_vec.reshape(blocksize_chroma, blocksize_chroma)
    Cb_block = Cb_vec.reshape(blocksize_chroma, blocksize_chroma)
    Cr_reconst = cv2.resize(Cr_block.astype(float), (blocksize, blocksize))
    Cb_reconst = cv2.resize(Cb_block.astype(float), (blocksize, blocksize))

    reconst_block = np.dstack([y_block, Cr_reconst, Cb_reconst])
    return reconst_block