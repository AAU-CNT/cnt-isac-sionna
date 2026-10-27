import numpy as np

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_USE_LEGACY_KERAS"] = "0"
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import datahandling
from VAE_RT import VAE_TF
# Define the similarity computation function

def compute_inv_mse(y_obs, y_hat, eps=1e-8):
    """
    Drop-in replacement for compute_sim (or compute_ssim_similarity).
    Computes inverse MSE: 1 / MSE(CIR_obs, CIR_hat).
    Higher value = more similar (exactly the same interface as your original functions).
    
    Args:
        y_obs: tf.Tensor [bo, height, width, 2]   (validation batch)
        y_hat: tf.Tensor [bh, height, width, 2]   (training batch)
        eps: small constant to avoid division by zero
    
    Returns:
        similarity: tf.Tensor [bo, bh]   (higher = better match)
    """
    # Broadcast both tensors so we get every validation point vs every training point
    y_obs_exp = tf.expand_dims(y_obs, axis=1)   # (bo, 1, H, W, 2)
    y_hat_exp = tf.expand_dims(y_hat, axis=0)   # (1, bh, H, W, 2)
    
    # Squared error over the complex CIR (real + imag channels)
    diff = y_obs_exp - y_hat_exp
    mse = tf.reduce_mean(tf.square(diff), axis=[2, 3, 4])   # mean over height, width, and 2 channels → (bo, bh)
    
    # Inverse MSE → higher value = more similar
    sim = 1.0 / (mse + eps)
    return sim

def compute_fingerprint_prob(y_obs, y_hat, sigma_db=2.0, dr_db=40.0):
    """
    Probabilistic RSS fingerprinting from Morais & Alkhateeb, "Localization in
    Digital Twin MIMO Networks" (arXiv:2403.09614).

    IMPORTANT: pass the RAW CIR here (do NOT run cir_to_pdp2 first). The paper's
    Gaussian model lives in the absolute-dB-power domain; the sum-to-1
    normalization in cir_to_pdp2 deletes the path-loss cue the method relies on.

    Each (AP, subcarrier) bin is one RSS measurement (analog of beam/subband
    (k, b) in the paper): RSS = |H|^2. RSS is taken to dB and floored dr_db below
    each sample's own peak, emulating receiver sensitivity instead of an
    arbitrary eps floor. Under the independence assumption (Eq. 11) the joint
    likelihood of a database position is the product of per-bin Gaussians
    (Eq. 12, std sigma_db); in log-domain that is -sum(dRSS_dB^2)/(2 sigma^2),
    so argmax reproduces the paper's estimate (Eq. 9) and higher = better match.

    Args:
        y_obs: tf.Tensor [bo, A, S, 2]   validation/user RAW CIR (real, imag)
        y_hat: tf.Tensor [bh, A, S, 2]   training/database RAW CIR (real, imag)
        sigma_db: RSS measurement std in dB (paper: 2 dBm)
        dr_db:    dynamic range; bins weaker than peak - dr_db are clipped

    Returns:
        similarity: tf.Tensor [bo, bh]   (higher = better match)
    """
    ln10 = tf.math.log(10.0)

    def to_db(y):
        # RSS = received power per (AP, subcarrier) bin, kept on its absolute scale
        p = tf.reduce_sum(tf.square(y), axis=-1)            # (b, A, S)
        db = 10.0 * tf.math.log(p + 1e-12) / ln10           # (b, A, S)
        # floor dr_db below each sample's peak (receiver sensitivity), not eps
        floor = tf.reduce_max(db, axis=[1, 2], keepdims=True) - dr_db
        return tf.maximum(db, floor)

    obs_db = to_db(y_obs)   # (bo, A, S)
    hat_db = to_db(y_hat)   # (bh, A, S)

    # Broadcast every val (user) sample against every train (database) sample
    diff = tf.expand_dims(obs_db, 1) - tf.expand_dims(hat_db, 0)   # (bo, bh, A, S)
    sq_err = tf.reduce_sum(tf.square(diff), axis=[2, 3])           # (bo, bh)

    # Log joint-likelihood (Gaussian product); shared constants dropped.
    return -sq_err / (2.0 * sigma_db ** 2)                         # (bo, bh)

def compute_sim(y_obs, y_hat):
    # y_obs: (bo, height, width, 2), y_hat: (bh, height, width, 2)
    # Expand dimensions for broadcasting
    obs_r = tf.expand_dims(y_obs[..., 0], 1)  # (bo, 1, height, width)
    obs_i = tf.expand_dims(y_obs[..., 1], 1)
    hat_r = tf.expand_dims(y_hat[..., 0], 0)  # (1, bh, height, width)
    hat_i = tf.expand_dims(y_hat[..., 1], 0)
    
    # Compute real and imaginary parts of the product
    real_part = obs_r * hat_r + obs_i * hat_i
    imag_part = obs_i * hat_r - obs_r * hat_i
    
    # Magnitude
    mag = tf.sqrt(real_part**2 + imag_part**2)  # (bo, bh, height, width)
    
    # Sum over height and width
    dot_both = tf.reduce_sum(mag, axis=[2, 3])  # (bo, bh)
    
    # Compute norms
    norm_obs = tf.sqrt(tf.reduce_sum(y_obs * y_obs, axis=[1, 2, 3]))  # (bo,)
    norm_hat = tf.sqrt(tf.reduce_sum(y_hat * y_hat, axis=[1, 2, 3]))  # (bh,)
    
    # Denominator
    denom = tf.expand_dims(norm_obs, 1) * tf.expand_dims(norm_hat, 0)  # (bo, bh)
    
    # Similarity (handle potential division by zero, though norms should be positive)
    sim = tf.where(denom > 0, dot_both / denom, tf.zeros_like(dot_both))
    
    return sim
def compute_ssim_similarity(y_obs, y_hat, **ssim_kwargs):
    """
    Compute batched SSIM-based similarity between two batches.
    
    Args:
        y_obs: tf.Tensor [bo, height, width, 2]   validation batch
        y_hat: tf.Tensor [bh, height, width, 2]   training batch
        **ssim_kwargs: passed to ssim_2d_rect (max_val, win_n, win_t, ...)

    Returns:
        similarity: tf.Tensor [bo, bh]   higher = more similar
    """
    # Your original function expects [B, n, T, C] — make sure shapes match
    # If your data is [B, height=n, width=T, C=2] then it's already compatible

    # We want every val sample vs every train sample → need to broadcast / tile
    bo = tf.shape(y_obs)[0]
    bh = tf.shape(y_hat)[0]

    # Tile validation to [bo*bh, n, T, 2]
    y_obs_tiled = tf.tile(
        tf.expand_dims(y_obs, 1),               # [bo, 1, n, T, 2]
        [1, bh, 1, 1, 1]
    )                                           # → [bo, bh, n, T, 2]
    y_obs_tiled = tf.reshape(y_obs_tiled, [-1, *y_obs.shape[1:]])   # [bo*bh, n, T, 2]

    # Tile prediction (training) accordingly
    y_hat_tiled = tf.tile(
        tf.expand_dims(y_hat, 0),
        [bo, 1, 1, 1, 1]
    )
    y_hat_tiled = tf.reshape(y_hat_tiled, [-1, *y_hat.shape[1:]])   # [bo*bh, n, T, 2]

    # Compute SSIM map for all pairs at once
    ssim_map = VAE_TF.ssim_2d_rect(           # assuming it's in VAE_TF class
        y_hat=y_hat_tiled,
        y_obs=y_obs_tiled,
        **ssim_kwargs
    )                                         # → [bo*bh, n, T, C] or similar

    # Reduce to one score per pair
    # Most common choices:
    #   - mean over all locations and channels
    score = tf.reduce_mean(ssim_map, axis=[1, 2, 3])   # [bo*bh]

    # Reshape back to pairwise matrix
    similarity = tf.reshape(score, [bo, bh])           # [bo, bh]

    return similarity
# Assuming you have your dataloader setup as provided
npz_path = "datasets/dense04_15AP.pkl"#"datasets/6_munich_15AP.pkl"#
dataloader = datahandling.dataloader(npz_path, train_split=0.8, a_tau_mode=False)
train_ds, test_ds, val_ds, tx_all, Dy, mean_h, std_h, channels, prior_dict = dataloader.make_dataset_shuffled(batch_size=512, train_snr_db=25,val_snr_db=25, FFT=False)

# List to hold results
results = []

# Loop over validation batches
for h_val, xyz_val, mask_val, id_val in val_ds:
    bo = tf.shape(h_val)[0].numpy()  # Batch size (may be smaller in last batch)
    
    # Initialize trackers for each sample in the val batch
    max_sims = [float('-inf')] * bo
    best_ids = [None] * bo
    best_xyzs = [None] * bo
    
    # Loop over train batches to find closest matches
    for h_train, xyz_train, mask_train, id_train in train_ds:
        #sim = compute_ssim_similarity(tf.squeeze(h_val), tf.squeeze(h_train))  # (bo, bh)
        # PDP-based methods (compute_sim / compute_inv_mse) use cir_to_pdp2;
        # compute_fingerprint_prob uses the RAW CIR. Keep both as locals so the
        # raw h_val/h_train are never overwritten across inner iterations.
        #h_train_pdp = dataloader.cir_to_pdp2(h_train)
        #h_val_pdp   = dataloader.cir_to_pdp2(h_val)
        sim = compute_sim(tf.squeeze(h_val), tf.squeeze(h_train))  # (bo, bh)
        #sim = compute_inv_mse(tf.squeeze(h_val_pdp), tf.squeeze(h_train_pdp))  # (bo, bh)
        #sim = compute_fingerprint_prob(tf.squeeze(h_val), tf.squeeze(h_train))  # (bo, bh) -- Morais & Alkhateeb DT fingerprinting (RAW CIR)

        # Find max sim and argmax for this train batch
        max_in_batch = tf.reduce_max(sim, axis=1).numpy()  # (bo,)
        argmax_in_batch = tf.argmax(sim, axis=1).numpy()  # (bo,)
        
        # Update for each val sample if better match found
        for i in range(bo):
            if max_in_batch[i] > max_sims[i]:
                max_sims[i] = max_in_batch[i]
                j = argmax_in_batch[i]
                best_ids[i] = id_train[j].numpy()  # Assuming id is convertible to numpy scalar/array
                best_xyzs[i] = xyz_train[j].numpy()  # Assuming xyz is a numpy-convertible array (e.g., shape (3,))
    
    #mu_post  = tf.reshape(best_xyzs[..., :2], [-1, 2])
    #xyz_true = tf.reshape(xyz_val[..., :2], [-1, 2])
    diff = best_xyzs - xyz_val.numpy()
    results.append(diff)
    import pickle
    with open('inference_clustering.pkl', 'wb') as f:
        pickle.dump({'mu': tf.squeeze(best_xyzs).numpy(),'z':tf.squeeze(best_xyzs).numpy(),'sigma':tf.zeros_like(xyz_val),'xyz':xyz_val.numpy()}, f)
results = np.array(results)
dist = np.linalg.norm(results, axis=-1)
median_dist = np.median(dist) 
print("mean", np.mean(dist))
print("median",median_dist)
print(tf.sqrt(tf.reduce_mean(tf.reduce_sum(results**2, axis=-1))))
