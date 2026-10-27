import numpy as np
import os
import tensorflow as tf
import pickle
import os
import re
from pathlib import Path
class dataloader:
    def __init__(self, path, train_split,a_tau_mode=False):
        self.path = path
        self.train_split = train_split
        with open(self.path, "rb") as f:
            h_complex, xyz, tx_xyz, _, _, _ = pickle.load(f)  # Load and unpack the tuple
        if len(h_complex.shape) == 2:
            h_complex= tf.expand_dims(h_complex, axis=1)

        self.APs=len(tx_xyz)
        self.dataset_len=len(h_complex)*train_split
        self.atau=a_tau_mode
    def read_data_file(self, ):
        with open(self.path, "rb") as f:
            h_complex, xyz, tx_xyz, a, tau, time_axis = pickle.load(f)  # Load and unpack the tuple
        if len(h_complex.shape) == 2:
            h_complex= tf.expand_dims(h_complex, axis=1)
        if self.atau:
            a_pad, tau_pad = dataloader.pad_paths (a, tau, 128, a[0][0][0].dtype, tau[0][0][0].dtype)
            h_all = tf.stack((tf.math.real(a_pad), tf.math.imag(a_pad), tau_pad), axis=-1)
            if tf.is_tensor(xyz):
                xyz = xyz.numpy()
            return h_all, xyz.astype(np.float32), tx_xyz.astype(np.float32)

        user_nr, AP_nr, time_index = h_complex.shape
        #user_nr, time_index = h_complex.shape
        h_complex_vector = tf.reshape(h_complex, (user_nr, -1))
        h_all = tf.stack((tf.math.real(h_complex_vector), tf.math.imag(h_complex_vector)), axis=-1)
        if tf.is_tensor(xyz):
            xyz = xyz.numpy()
        return h_all, xyz.astype(np.float32), tx_xyz.astype(np.float32)

    
    def make_dataloaders(self, h, xyz, mask, idx, batch_size,
                     mean_h, std_h, snr_db=None,
                     dataset_signal_power=None):
        AUTOTUNE = tf.data.AUTOTUNE
        ds = tf.data.Dataset.from_tensor_slices((h, xyz, mask, idx))
        ds = ds.shuffle(
            buffer_size=min(int(h.shape[0]), 50000),
            reshuffle_each_iteration=True
        )
        if snr_db is None:
            ds = ds.map(
                lambda h, xyz, mask, idx: self.scale_only(
                    h, xyz, mask, idx, mean_h, std_h
                ),
                num_parallel_calls=AUTOTUNE,
                deterministic=True
            )
        else:
            ds = ds.map(
                lambda h, xyz, mask, idx: dataloader.add_noise_then_scale_dataset(
                    h, xyz, mask, idx, mean_h, std_h, snr_db, dataset_signal_power
                ),
                num_parallel_calls=AUTOTUNE,
                deterministic=True
            )
        ds = ds.batch(batch_size, drop_remainder=True)
        ds = ds.prefetch(AUTOTUNE)
        return ds

    
    def split_grouped_tensor(self, h, N):
        dy_total = tf.shape(h)[1]
        remainder = tf.math.floormod(dy_total, N)

        # Runtime check: Dy_total must be divisible by N
        with tf.control_dependencies([
            tf.debugging.assert_equal(
                remainder,
                0,
                message="Dy_total must be divisible by N."
            )
        ]):
            dy = dy_total // N
            # reshape: [num_samples, Dy_total, 2] -> [num_samples, N, Dy, 2]
            h_grouped = tf.reshape(h, [-1, N, dy, tf.shape(h)[-1]])

        return h_grouped


    def make_dataset_shuffled(self, batch_size, train_snr_db=40.0, val_snr_db=40.0, FFT=False):
        h_all, xyz_all, tx_all = self.read_data_file()
        print(tx_all)
        mini_bs=len(tx_all)
        N = h_all.shape[0]
        xyz_all=xyz_all[:N]
        idx_all = tf.range(N)
        # --- shuffle before splitting ---
        
        perm = tf.random.shuffle(tf.range(N), seed=42)   # shape [N], dtype int32
        h_all  = tf.gather(h_all, perm, axis=0)
        xyz_all = tf.gather(xyz_all, perm, axis=0)
        idx_all = tf.gather(idx_all, perm, axis=0)

        # --- split ---
        train_cut = int(self.train_split * N)
        test_cut = int(1.5*(1-self.train_split)/2 * N + train_cut)
        h_tr, h_test, h_va = h_all[:train_cut], h_all[train_cut:test_cut], h_all[test_cut:N]
        xyz_tr, xyz_test, xyz_va = xyz_all[:train_cut], xyz_all[train_cut:test_cut], xyz_all[test_cut:N]
        idx_tr, idx_test, idx_va = idx_all[:train_cut], idx_all[train_cut:test_cut], idx_all[test_cut:N]
        h_tr = self.split_grouped_tensor(h_tr, mini_bs)
        h_test = self.split_grouped_tensor(h_test, mini_bs)
        h_va = self.split_grouped_tensor(h_va, mini_bs)
        Dy = h_tr.shape[2]
        self.CIR_mean, self.CIR_std=self.compute_perN_stats(h_tr)
        
        if FFT:
            h_tr = dataloader.fft_time_axis(h_tr)
            h_test= dataloader.fft_time_axis(h_test)
            h_va = dataloader.fft_time_axis(h_va)
            self.CFR_mean, self.CFR_std=self.compute_perN_stats(h_tr)

        mask_tr = self.per_N_is_close_to_zero(h_tr, atol=1e-8)
        mask_test = self.per_N_is_close_to_zero(h_test, atol=1e-8)
        mask_va = self.per_N_is_close_to_zero(h_va, atol=1e-8)
        mean_h, std_h = self.compute_perN_stats(h_tr)
        dataset_signal_power = self.compute_dataset_signal_power(h_tr, mask_tr)
        self.dataset_signal_power = dataset_signal_power

        train_ds = self.make_dataloaders(
            h_tr, xyz_tr, mask_tr, idx_tr, batch_size,
            mean_h=mean_h, std_h=std_h, snr_db=train_snr_db,
            dataset_signal_power=dataset_signal_power
        )

        test_ds = self.make_dataloaders(
            h_test, xyz_test, mask_test, idx_test, batch_size,
            mean_h=mean_h, std_h=std_h, snr_db=train_snr_db,
            dataset_signal_power=dataset_signal_power
            
        )

        self._val_raw = (h_va, xyz_va, mask_va, idx_va)
        self._val_stats = (mean_h, std_h)

        val_ds = self.make_dataloaders(
            h_va, xyz_va, mask_va, idx_va, batch_size,
            mean_h=mean_h, std_h=std_h, snr_db=val_snr_db,
            dataset_signal_power=dataset_signal_power
        )
        h0, xyz0, mask0, idb0 = next(iter(train_ds))
        prior_dict = self.make_prior_dict(xyz0[0], h0[0])
        prior_dict['signal_P']=self.dataset_signal_power
        return train_ds, test_ds, val_ds,tx_all, Dy, mean_h, std_h, 2, prior_dict
    
    
    def rebuild_val_ds(self, mode, los_csv_path, batch_size, snr_db):
        """Return a new val dataset filtered by LoS mode ("All", "LoS", "NLoS")."""
        h_va, xyz_va, mask_va, idx_va = self._val_raw
        mean_h, std_h = self._val_stats

        total = len(h_va) if hasattr(h_va, "__len__") else h_va.shape[0]
        if mode != "All":
            los_xy = np.loadtxt(los_csv_path, delimiter=",")[:, :2]
            los_set = set(map(tuple, np.round(los_xy, 1).tolist()))

            xyz_np = xyz_va.numpy() if hasattr(xyz_va, "numpy") else np.array(xyz_va)
            in_los = np.array([
                (round(float(r[0]), 1), round(float(r[1]), 1)) in los_set
                for r in xyz_np
            ])
            keep = in_los if mode == "LoS" else ~in_los

            n_keep = int(keep.sum())

            h_va_np = h_va.numpy() if hasattr(h_va, "numpy") else np.array(h_va)
            mask_va_np = mask_va if isinstance(mask_va, np.ndarray) else np.array(mask_va)
            idx_va_np = idx_va.numpy() if hasattr(idx_va, "numpy") else np.array(idx_va)

            h_va    = h_va_np[keep]
            xyz_va  = xyz_np[keep]
            mask_va = mask_va_np[keep]
            idx_va  = idx_va_np[keep]

            if n_keep < batch_size:
                repeats = -(-batch_size // n_keep)  # ceil division
                h_va    = np.concatenate([h_va]    * repeats, axis=0)
                xyz_va  = np.concatenate([xyz_va]  * repeats, axis=0)
                mask_va = np.concatenate([mask_va] * repeats, axis=0)
                idx_va  = np.concatenate([idx_va]  * repeats, axis=0)
                print(f"val_mode={mode}: {n_keep}/{total} unique samples kept "
                      f"({100*n_keep/total:.1f}%), oversampled ×{repeats} → "
                      f"{len(h_va)//batch_size} full batches of {batch_size}")
            else:
                print(f"val_mode={mode}: {n_keep}/{total} samples kept "
                      f"({100*n_keep/total:.1f}%) → {n_keep//batch_size} full batches of {batch_size}")
        else:
            print(f"val_mode=All: {total} samples → {total // batch_size} full batches of {batch_size}")

        return self.make_dataloaders(
            h_va, xyz_va, mask_va, idx_va,
            batch_size, mean_h, std_h, snr_db
        )

    def find_latest(self, folder, prefix: str, extension: str = ".keras"):
        """Return (latest_file_path, latest_number, next_file_path)."""

        #folder = Path(__file__).parent
        pattern = re.compile(rf"^{re.escape(prefix)}_(\d+){re.escape(extension)}$")

        latest_num = -1
        latest_file = "model"

        for file in os.listdir(folder):
            m = pattern.match(file)
            if m:
                num = int(m.group(1))
                if num > latest_num:
                    latest_num = num
                    latest_file = file

        # If no file existed, start from 0
        next_num = latest_num + 1
        next_file = folder + f"{prefix}_{next_num}{extension}"

        return folder+latest_file, latest_num, next_file
    


    def make_prior_dict(self, xyz, h):
        """
        Build a simple prior dictionary:
        - parameter_dimensions = dim of xyz
        - obs_dimensions = dim of h (second axis)
        """
        xyz=xyz[:2]
        parameter_dimensions = xyz.shape[-1]
        obs_dimensions = h.shape[1]

        # Prior mean
        prior_mean_diag = 0.0
        prior_mean = prior_mean_diag * np.ones(parameter_dimensions, dtype=np.float32)
        rand_matrix = np.random.normal(0,1,
                        size=(parameter_dimensions, parameter_dimensions))
        prior_covariance = (np.finfo(np.float32).eps*np.identity(parameter_dimensions) +\
                        np.matmul(rand_matrix, np.transpose(rand_matrix))).astype(np.float32)
        prior_covariance_inverse = np.linalg.inv(prior_covariance).astype(np.float32)
        prior_covariance_cholesky = np.linalg.cholesky(prior_covariance)
        prior_covariance_cholesky_inverse = np.linalg.inv(prior_covariance_cholesky)

        # Observation-side noise regularization
        noise_regularization_matrix = np.ones((1, obs_dimensions), dtype=np.float32)
        wm = tf.convert_to_tensor(noise_regularization_matrix, dtype=tf.float32)
        wm = tf.reshape(wm, (1, 1, obs_dimensions, 1))
        wm = tf.repeat(wm, repeats=2, axis=-1)

        prior_dict = {
            "prior_mean": prior_mean,
            "prior_covariance": prior_covariance,
            "prior_covariance_inverse": prior_covariance_inverse,
            "prior_covariance_cholesky": prior_covariance_cholesky,
            "prior_covariance_cholesky_inverse": prior_covariance_cholesky_inverse,
            "noise_regularization_matrix": wm,
            "parameter_dimensions": parameter_dimensions,
            "obs_dimensions": obs_dimensions,
        }

        return prior_dict
    
    def compute_global_mean_std(self, H):
        """
        H: [batch, N, 1016, 2]
        Returns:
            mean: (2,)  tensor with mean for real and imag
            std:  (2,)  tensor with std  for real and imag
        """

        # Flatten everything *except* the last dimension (real/imag)
        # Result: [num_total_values, 2]
        H_flat = tf.reshape(H, [-1, 2])

        mean = tf.reduce_mean(H_flat, axis=0)  # shape (2,)
        std  = tf.math.reduce_std(H_flat, axis=0) + 1e-12  # avoid div by zero

        return mean, std
    
    def compute_perN_stats(self, h):
        """
        h has shape (batch, N, T, C)
        We compute mean/std over batch and T (time), for each N and each channel.
        Output shapes:
            mean_h: (1, N, 1, C)
            std_h:  (1, N, 1, C)
        """
        # reduce over batch and time only: axes 0 and 2
        mean_h = tf.reduce_mean(h, axis=(0, 2), keepdims=True)   # -> (1, N, 1, C)
        std_h  = tf.math.reduce_std(h, axis=(0, 2), keepdims=True)

        # avoid division by zero
        std_h = tf.where(std_h == 0.0, tf.ones_like(std_h), std_h)

        return mean_h, std_h
    @staticmethod
    def scale(h, mean, std):
        h_scaled=(h - mean) / std
        return h_scaled
    @staticmethod
    def unscale(h, mean, std):
        h_unscaled = h * std + mean
        return h_unscaled
    
    @staticmethod
    def fft_time_axis(h):
        real = h[..., 0]
        imag = h[..., 1]
        h_complex = tf.complex(real, imag)
        H_complex = tf.signal.fft(h_complex) 
        H = tf.stack(
            [tf.math.real(H_complex),
            tf.math.imag(H_complex)],
            axis=-1
        )
        return H
    @staticmethod
    def ifft_time_axis(H):
        """
        Inverse FFT along the time/frequency axis.
        Expects input shape:  [batch, APs, N_subcarriers, 2]  (real + imag)
        Returns shape:        [batch, APs, N_time_samples, 2] (same outer dims)
        """
        # Reconstruct complex tensor
        real = H[..., 0]
        imag = H[..., 1]
        H_complex = tf.complex(real, imag)

        # Inverse FFT along the frequency axis (axis=-2)
        h_complex = tf.signal.ifft(H_complex)

        # Back to real/imag channels
        h = tf.stack(
            [tf.math.real(h_complex),
            tf.math.imag(h_complex)],
            axis=-1
        )
        return h
    def per_N_is_close_to_zero(self, x, atol=1e-8, rtol=0.0):
        zero_mask_bn = np.all(
            np.isclose(np.abs(x), 0.0, atol=1e-8),
            axis=(2, 3)
        )
        return ~zero_mask_bn
    def add_noise(self, h, xyz, mask, idx, sigma=0.01):
        noise = tf.random.normal(
            shape=tf.shape(h),
            mean=0.0,
            stddev=sigma,
            dtype=h.dtype
        )
        return h + noise, xyz, mask, idx
    
    
    @staticmethod
    def pad_paths(a_list, tau_list, P_fixed=128,
              a_dtype=tf.complex64, tau_dtype=tf.float32):

        # Stack into ragged tensors along batch dimension
        a_rt   = tf.ragged.stack([tf.convert_to_tensor(tf.squeeze(x), dtype=a_dtype)[..., :P_fixed]
                                for x in a_list])     # (B, 8, <=P_fixed)
        tau_rt = tf.ragged.stack([tf.convert_to_tensor(tf.squeeze(x), dtype=tau_dtype)[..., :P_fixed]
                                for x in tau_list])   # (B, 8, <=P_fixed)

        # Convert to dense by padding/truncating to fixed length P_fixed
        a_pad   = a_rt.to_tensor(default_value=tf.cast(0, a_dtype),
                                shape=[None, None, P_fixed])        # (B, 8, P_fixed)
        tau_pad = tau_rt.to_tensor(default_value=tf.cast(0.0, tau_dtype),
                                shape=[None, None, P_fixed])      # (B, 8, P_fixed)

        return a_pad, tau_pad
    
    def scale_only(self, h, xyz, mask, idx, mean_h, std_h):
        #h = tf.cast(h, tf.float32)
        #mean_h = tf.cast(mean_h, h.dtype)
        #std_h = tf.cast(std_h, h.dtype)

        h_scaled = (h - mean_h) / std_h
        return h_scaled, xyz, mask, idx
    
    @staticmethod
    def compute_dataset_signal_power(h, mask):
        """
        Mean of h^2 over active entries (mask == 1) across the whole training set.
        
        h:    [num_samples, N, Dy, C]   (after FFT if applicable)
        mask: [num_samples, N]          (1 for active, 0 for inactive)
        
        Returns: scalar tf.float32
        """
        h = tf.cast(h, tf.float32)
        mask_bc = tf.cast(mask, tf.float32)

        # Broadcast mask up to h's rank (e.g. [num_samples, N] -> [num_samples, N, Dy, C])
        while len(mask_bc.shape) < len(h.shape):
            mask_bc = mask_bc[..., None]
        mask_bc = tf.broadcast_to(mask_bc, tf.shape(h))

        sum_sq     = tf.reduce_sum(tf.square(h) * mask_bc)
        num_active = tf.reduce_sum(mask_bc)
        return sum_sq / tf.maximum(num_active, 1.0)

    @staticmethod
    def add_noise_then_scale_dataset(h, xyz, mask, idx, mean_h, std_h,
                                    snr_db, dataset_signal_power):
        """
        Add i.i.d. Gaussian noise with a FIXED variance derived from the
        dataset-average signal power. SNR target is met on average across the
        dataset, not exactly per sample. Noise is masked to active anchors only.
        
        h:                    [N, Dy, C]
        mask:                 [N]
        mean_h/std_h:         broadcastable, typically [1, N, 1, C]
        dataset_signal_power: scalar, precomputed once over the training set
                            (mean of h^2 over active entries, see helper below)
        """
        # Broadcast mask from [N] -> [N, Dy, C]
        mask_bc = tf.cast(mask[:, None, None], h.dtype)
        mask_bc = tf.broadcast_to(mask_bc, tf.shape(h))

        # Fixed noise std from dataset-level signal power -- no per-sample stats
        snr_linear = tf.pow(tf.constant(10.0, dtype=h.dtype), snr_db / 10.0)
        sigma = tf.sqrt(dataset_signal_power / snr_linear)

        # i.i.d. Gaussian noise, masked to active anchors
        noise = tf.random.normal(tf.shape(h), mean=0.0, stddev=sigma, dtype=h.dtype)
        noise = noise * mask_bc

        h_noisy  = h + noise
        h_scaled = (h_noisy - mean_h) / std_h
        return h_scaled, xyz, mask, idx


    @staticmethod
    def add_noise_then_scale_dataset_batch(h, xyz, mask, idx, mean_h, std_h,
                                            snr_db, dataset_signal_power):
        """
        Batched version. h: [B, A, Dy, C], mask: [N] or [B, A].
        """
        h_shape     = tf.shape(h)              # [B, A, Dy, C]
        B           = h_shape[0]
        num_anchors = h_shape[1]

        # Graph-safe handling of mask shape (matches your existing pattern)
        mask = tf.cond(
            tf.equal(tf.rank(mask), 1),
            lambda: tf.broadcast_to(mask[:, None], [B, num_anchors]),
            lambda: mask
        )

        # Broadcast to full shape [B, A, Dy, C]
        mask_bc = tf.cast(mask[:, :, None, None], h.dtype)
        mask_bc = tf.broadcast_to(mask_bc, h_shape)

        # Fixed sigma from dataset-level stat
        snr_linear = tf.pow(tf.constant(10.0, dtype=h.dtype), snr_db / 10.0)
        sigma = tf.sqrt(dataset_signal_power / snr_linear)

        # i.i.d. Gaussian noise, masked
        noise = tf.random.normal(h_shape, mean=0.0, stddev=sigma, dtype=h.dtype)
        noise = noise * mask_bc

        h_noisy  = h + noise
        h_scaled = (h_noisy - mean_h) / std_h
        return h_scaled, xyz, mask, idx


    @staticmethod
    def per_sample_add_noise_then_scale(h, xyz, mask, idx, mean_h, std_h, snr_db):
        """
        Add Gaussian noise to the *unnormalized* sample h so that the realized
        SNR matches snr_db over active anchors only, then normalize with mean/std.

        h:    [N, Dy, C]
        mask: [N]
        mean_h/std_h: broadcastable, typically [1, N, 1, C] from compute_perN_stats
        """
        #mean_h = tf.cast(mean_h, h.dtype)
        #std_h = tf.cast(std_h, h.dtype)
        #snr_db = tf.cast(snr_db, h.dtype)

        # Broadcast mask from [N] -> [N, Dy, C]

        mask_bc = tf.cast(mask[:, None, None], h.dtype)
        mask_bc = tf.broadcast_to(mask_bc, tf.shape(h))

        # Signal power over active entries only
        num_active = tf.reduce_sum(mask_bc)
        num_active = tf.maximum(num_active, tf.constant(1.0, dtype=h.dtype))

        signal_power = tf.reduce_sum(tf.square(h) * mask_bc) / num_active

        # Convert SNR(dB) -> target noise power
        snr_linear = tf.pow(tf.constant(10.0, dtype=h.dtype), snr_db / 10.0)
        target_noise_power = signal_power / snr_linear

        # Draw unit Gaussian noise and restrict it to active anchors
        noise = tf.random.normal(tf.shape(h), mean=0.0, stddev=1.0, dtype=h.dtype)
        noise = noise * mask_bc

        # Rescale the realization so realized noise power matches the target exactly
        raw_noise_power = tf.reduce_sum(tf.square(noise)) / num_active
        raw_noise_power = tf.maximum(raw_noise_power, tf.constant(1e-12, dtype=h.dtype))

        scale = tf.sqrt(target_noise_power / raw_noise_power)
        noise = noise * scale

        h_noisy = h + noise
        h_scaled = (h_noisy - mean_h) / std_h
        return h_scaled, xyz, mask, idx

    @staticmethod
    def per_sample_add_noise_then_scale_batch(h, xyz, mask, idx, mean_h, std_h, snr_db):
        
        h_shape = tf.shape(h)          # [B, A, Dy, C]
        B = h_shape[0]
        num_anchors = h_shape[1]

        # === Graph-safe handling of mask shape ===
        mask = tf.cond(
            tf.equal(tf.rank(mask), 1),
            lambda: tf.broadcast_to(mask[:, None], [B, num_anchors]),
            lambda: mask
        )
        # Now mask is guaranteed [B, num_anchors]

        # Broadcast to full shape [B, num_anchors, Dy, C]
        mask_bc = tf.cast(mask[:, :, None, None], h.dtype)
        mask_bc = tf.broadcast_to(mask_bc, h_shape)

        # Signal power over active entries only
        num_active = tf.reduce_sum(mask_bc)
        num_active = tf.maximum(num_active, tf.constant(1.0, dtype=h.dtype))
        signal_power = tf.reduce_sum(tf.square(h) * mask_bc) / num_active

        # Convert SNR(dB) → target noise power
        snr_linear = tf.pow(tf.constant(10.0, dtype=h.dtype), snr_db / 10.0)
        target_noise_power = signal_power / snr_linear

        # Draw unit Gaussian noise and restrict to active anchors
        noise = tf.random.normal(h_shape, mean=0.0, stddev=1.0, dtype=h.dtype)
        noise = noise * mask_bc

        # Rescale the realization so realized noise power matches the target exactly
        raw_noise_power = tf.reduce_sum(tf.square(noise)) / num_active
        raw_noise_power = tf.maximum(raw_noise_power, tf.constant(1e-12, dtype=h.dtype))
        scale = tf.sqrt(target_noise_power / raw_noise_power)
        noise = noise * scale

        h_noisy = h + noise
        h_scaled = (h_noisy - mean_h) / std_h

        return h_scaled, xyz, mask, idx 

    @staticmethod
    def cir_to_pdp2(y):
        # y: (..., height, width, 2)
        pdp = tf.reduce_sum(tf.square(y), axis=-1)  # (..., height, width)
        # optional but recommended: normalize per example so power doesn't dominate
        pdp = pdp / (tf.reduce_sum(pdp, axis=[-2, -1], keepdims=True) + 1e-12)
        return tf.stack([pdp, tf.zeros_like(pdp)], axis=-1)  # (..., height, width, 2)