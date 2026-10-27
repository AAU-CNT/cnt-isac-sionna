import numpy as np
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_USE_LEGACY_KERAS"] = "0"
import tensorflow as tf
seed = 20
tf.random.set_seed(seed)
tf.keras.utils.set_random_seed(seed)
np.random.seed(seed)
tf.get_logger().setLevel('ERROR')
import datahandling
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
#from VAE_RT import VAE_TF
# === EXACT ORIGINAL compute_sim (no changes) ===

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


def gaussian_1d(size, sigma, dtype=tf.float32):
        r = tf.range(-(size//2), size//2 + 1, dtype=dtype)
        g = tf.exp(-(r*r) / (2.0 * sigma*sigma))
        return g / tf.reduce_sum(g)
def ssim_2d_rect(y_hat, y_obs, max_val=1.0,
                    win_n=3, win_t=11, sigma_n=1.0, sigma_t=1.5, eps=1e-12):
        """
        Rectangular-window SSIM over (n, T) with window win_n x win_t.
        y_hat, y_obs: [B, n, T, C] (C=2 ok)
        Returns scalar mean SSIM.
        """
        
        x = y_hat
        y = y_obs
        B = tf.shape(x)[0]
        n = tf.shape(x)[1]
        T = tf.shape(x)[2]
        C = tf.shape(x)[3]

        # Build separable 2D depthwise kernel: k_n (over n) and k_t (over T)
        g_n = gaussian_1d(win_n, sigma_n, dtype=x.dtype)  # [win_n]
        g_t = gaussian_1d(win_t, sigma_t, dtype=x.dtype)  # [win_t]
        k2d = tf.tensordot(g_n, g_t, axes=0)              # [win_n, win_t]
        k2d = k2d / tf.reduce_sum(k2d)

        # Depthwise conv2d expects kernel [kh, kw, in_ch, channel_multiplier]
        k = tf.reshape(k2d, [win_n, win_t, 1, 1])         # [win_n, win_t, 1, 1]
        k = tf.tile(k, [1, 1, C, 1])                      # [win_n, win_t, C, 1]

        def filt(z):
            return tf.nn.depthwise_conv2d(z, k, strides=[1,1,1,1], padding="SAME")

        mu_x = filt(x)
        mu_y = filt(y)

        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_x2 = filt(x*x) - mu_x2
        sigma_y2 = filt(y*y) - mu_y2
        sigma_x2 = tf.maximum(sigma_x2, 0.0)
        sigma_y2 = tf.maximum(sigma_y2, 0.0)
        sigma_xy = filt(x*y) - mu_xy

        K1, K2 = 0.05, 0.15
        C1 = (K1 * max_val)**2
        C2 = (K2 * max_val)**2

        num = (2.0*mu_xy + C1) * (2.0*sigma_xy + C2)
        den_safe = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2) + 1e-10
        ssim_map = num / den_safe
        ssim_map = tf.where(tf.math.is_nan(ssim_map) | tf.math.is_inf(ssim_map), 0.0, ssim_map)
        return ssim_map
def compute_ssim(y_obs, y_hat, **ssim_kwargs):
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
    ssim_map = ssim_2d_rect(           # assuming it's in VAE_TF class
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


def compute_sim(y_obs, y_hat):
    obs_r = tf.expand_dims(y_obs[..., 0], 1)
    obs_i = tf.expand_dims(y_obs[..., 1], 1)
    hat_r = tf.expand_dims(y_hat[..., 0], 0)
    hat_i = tf.expand_dims(y_hat[..., 1], 0)

    real_part = obs_r * hat_r + obs_i * hat_i
    imag_part = obs_i * hat_r - obs_r * hat_i
    mag = tf.sqrt(real_part**2 + imag_part**2)
    dot_both = tf.reduce_sum(mag, axis=[2, 3])

    norm_obs = tf.sqrt(tf.reduce_sum(y_obs * y_obs, axis=[1, 2, 3]))
    norm_hat = tf.sqrt(tf.reduce_sum(y_hat * y_hat, axis=[1, 2, 3]))

    denom = tf.expand_dims(norm_obs, 1) * tf.expand_dims(norm_hat, 0)
    sim = tf.where(denom > 0, dot_both / denom, tf.zeros_like(dot_both))
    return sim

# === LOAD DATA ===
npz_path = "datasets/6_munich_5AP.pkl"
dataloader = datahandling.dataloader(npz_path, train_split=0.8, a_tau_mode=False)
train_ds, test_ds, val_ds, tx_all, Dy, mean_h, std_h, channels, prior_dict = dataloader.make_dataset_shuffled(batch_size=256, FFT=False)

# === SELECT ONE VALIDATION POINT ===
val_iterator = iter(val_ds)
h_val_batch, xyz_val_batch, mask_val_batch, id_val_batch = next(val_iterator)

val_idx = 0   # ← change this if you want another sample

chosen_raw = tf.squeeze(h_val_batch[val_idx])
chosen_h_val = tf.expand_dims(chosen_raw, 0)
chosen_xyz = xyz_val_batch[val_idx].numpy()

print(f"Chosen validation point (index {val_idx}) at position: {chosen_xyz}")
print(f"Final chosen CIR shape: {chosen_h_val.shape}")

# === COMPUTE SIMILARITY TO EVERY TRAINING POINT ===
all_similarities = []
all_xyz = []

print("Computing similarity of chosen point to every training CIR...")
for h_train_batch, xyz_train_batch, mask_train_batch, id_train_batch in train_ds:
    h_train_clean = tf.squeeze(h_train_batch)
    sim_batch = compute_inv_mse(chosen_h_val, h_train_clean)
    #sim_batch = compute_sim(chosen_h_val, h_train_clean)
    #sim_batch = compute_ssim(chosen_h_val, h_train_clean)
    sim_flat = tf.squeeze(sim_batch).numpy()
    
    all_similarities.append(sim_flat)
    all_xyz.append(xyz_train_batch.numpy())

all_similarities = np.concatenate(all_similarities)
all_xyz = np.concatenate(all_xyz, axis=0)

print(f"Done! Processed {len(all_similarities)} training points.")
print(f"Similarity range → min: {all_similarities.min():.4f} | max: {all_similarities.max():.4f}")

# === COLOR BY RANKING (0 = least similar → 1 = most similar) ===
rank_order = np.argsort(all_similarities)
normalized_ranking = np.zeros_like(all_similarities)
normalized_ranking[rank_order] = np.linspace(0.0, 1.0, len(all_similarities))

gamma = 4.0                          # ← TUNE THIS (4.0 is very strong, 3.0 is still good, 2.5 is milder)
norm = PowerNorm(gamma=gamma, vmin=0.0, vmax=1.0)

print(f"Using non-linear gamma = {gamma} → almost no color change 0–0.5, full resolution 0.5–1.0")

# === PLOT: COLORED SCATTER (heatmap-style) — no interpolation, empty areas stay blank ===
fig, ax = plt.subplots(figsize=(13, 5))

scatter = ax.scatter(
    all_xyz[:, 0],
    all_xyz[:, 1],
    c=normalized_ranking,
    norm=norm,
    cmap='magma', 
    s=16,
    alpha=0.92,
    edgecolors='none'
)

# Chosen validation point (big red X on top)
ax.scatter(
    chosen_xyz[0],
    chosen_xyz[1],
    marker='X',
    color='red',
    s=256,
    linewidths=4,
    label='Chosen Validation Point',
    zorder=10,
    edgecolors='white'
)

cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
cbar.set_ticks([0.0, 0.6, 0.8, 1.0])
cbar.set_label('Similarity Ranking', fontsize=8)#\n(White/Yellow = most similar  •  Dark Purple/Black = least similar)', fontsize=8)

ax.set_xlabel('X Position (m)', fontsize=8)
ax.set_ylabel('Y Position (m)', fontsize=8)
ax.set_title('MSE Similarity Heatmap \n Chosen Validation Point vs. All Training Points', fontsize=8)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

plt.tight_layout()
#plt.show()

# Optional: save high-resolution version
plt.savefig('mse_cir_similarity_heatmap_points.pdf', dpi=400, bbox_inches='tight')