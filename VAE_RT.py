import warnings
import os

# Silence Python warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow import keras
from tensorflow.keras import layers
import pickle

import numpy as np
import raytracer as rt
import json
import time
from cnn_model import build_CNN_encoder, build_parallel_CNN_encoder
from transformer_model import AnchorTransformerEncoder
#tf.config.run_functions_eagerly(True)

from datahandling import dataloader

class VAE_TF(keras.Model):
    def __init__(self, batches, num_anchors, input_dim, latent_dim, channels=2,use_fft=False, 
                 RT_snr_db=0, scaler_mean=None, scaler_std=None, tx=None, prior_dict=None, 
                 a_tau_mode=False, Baseline_mode=False, MAG_inference=False,model_type='transformer'):
        super().__init__()
        self.scenario = 'munich'  # 'munich' | 'AAU'
        self.device="/GPU:0" if tf.test.is_built_with_cuda() else "/CPU:0"
        print("USING", self.device)
        self.Use_FFT = use_fft
        self.atau_mode=a_tau_mode
        self.Baseline_mode=Baseline_mode
        self.use_MAG_inference=MAG_inference
        self.tx=tx
        self.model_type=model_type
        self.prior_dict = prior_dict
        self.batches=batches
        self.num_anchors = num_anchors
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.z_const = self.add_weight(
            name="z_height",
            shape=(),
            initializer=tf.constant_initializer(1.15),
            trainable=False   # or False if you want to freeze it
        )
        print(num_anchors)

        self.prior_headers=[]
        if self.model_type=='Transformer':
            #self.encoder = AnchorTransformerEncoder(N_anchors=num_anchors,cir_len=input_dim,
            #                                        in_ch=channels,embed_dim=384,num_heads=8,mlp_dim=512,
            #                                        num_blocks=2,dropout=0.05,weight_decay=1e-5,out_dim=1024,)
            self.encoder = AnchorTransformerEncoder(N_anchors=num_anchors,cir_len=input_dim,
                                                    in_ch=channels,embed_dim=384,num_heads=8,mlp_dim=1024,
                                                    num_blocks=4,dropout=0.05,weight_decay=1e-5,out_dim=1024,)
        elif self.model_type=='CNN':
            self.encoder = keras.Sequential([
                build_CNN_encoder(input_shape=(num_anchors, input_dim, 2)),
            ])
        elif self.model_type=='UWB':
            # Paper-exact: K parallel CNNs over consecutive 3-anchor subsets, each
            # emitting its own (x, y). Outputs are stacked here as (batch, K, latent_dim)
            # and averaged by the head below (the paper's K-model ensemble averaging).
            if not self.Baseline_mode:
                raise ValueError(
                    "model_type='UWB' is a deterministic per-anchor-subset regressor and "
                    "only supports Baseline_mode=True; the un-averaged (batch, K, latent_dim) "
                    "encoder output is not compatible with the VAE mu/logvar heads."
                )
            self.encoder = keras.Sequential([
                build_parallel_CNN_encoder(input_shape=(num_anchors, input_dim, 2),
                                           out_dim=latent_dim, average=False),
            ])
        if self.Baseline_mode and self.model_type=='UWB':
            # The "head" is just the averaging of the K per-model predictions ->
            # a single (batch, latent_dim) position, keeping downstream code unchanged.
            self.prior_headers.append(
                layers.GlobalAveragePooling1D(name="mu_head")
            )
        elif self.Baseline_mode:
            self.prior_headers.append(
                layers.Dense(latent_dim, name=f"mu_head")
            )
        else:
            self.num_u_draw=3 #Please touch this variable
            print("Draws:", self.num_u_draw)
            self.sub_models = 1 #Don't touch this variable plz
            for i in range(self.sub_models):
                self.prior_headers.append([
                    layers.Dense(latent_dim, name=f"mu_head_{i}"),           # mu
                    layers.Dense(latent_dim, name=f"logvar_head_{i}") 
                    #layers.Dense(latent_dim+1, name=f"logvar_head_{i}") 
                ])
            self.bandwidth=100e6
            
            self.time_axis=np.arange(start=0, stop=2e-6, step=1/self.bandwidth)
            if self.scenario == 'AAU':
                self.ch, rxPos_grid, txPos = rt.create_channel_AAU()
            elif self.scenario == 'munich':
                self.ch, rxPos_grid, txPos = rt.create_channel_munich(txPos=self.tx)
            elif self.scenario == 'qorvo':
                
                # Obtained from DW1000 User Manual section on CIR and Table 8 of the DW3000 user manual.
                num_samples=1016
                fundamental_frequency = 499.2 * 1e6
                center_frequency = 6489.6 * 1e6 
                time_step = 1 / (fundamental_frequency * 2)
                self.time_axis = np.arange(0, num_samples) * time_step
                self.freq_axis = np.fft.fftfreq(num_samples, time_step) + center_frequency
                self.bandwidth = 499.2 * 1e6 #* 1.3 #1.3 to make it 30% larger
                
                self.ch, rxPos_grid, txPos = rt.create_channel_qorvo(center_frequency, txPos=self.tx)
            else:
                raise ValueError('unknown scene')
            self.ch.prep_scene(batches)
            self.bwcir=rt.bandlimiter(self.bandwidth)
        self.RT_snr_db=RT_snr_db
        self.scaler_mean = scaler_mean
        self.scaler_std  = scaler_std
        self.prior_dict['penalty_js']=0.5
        self.prior_dict['identity_otimes_prior_cov_inv'] =\
            tf.linalg.LinearOperatorKronecker(
                    [tf.linalg.LinearOperatorFullMatrix(tf.eye(self.prior_dict['parameter_dimensions'])),
                    tf.linalg.LinearOperatorFullMatrix(self.prior_dict['prior_covariance_inverse'])])

    
    @tf.function(jit_compile=False)
    def encode(self, x, mask, training):
        mu_list = []
        logvar_list = []
        if self.model_type=='Transformer':
            h = self.encoder(x, mask, training=training)
        elif self.model_type=='CNN' or self.model_type=='UWB':
            h = self.encoder(tf.squeeze(x), training=training)
        if self.Baseline_mode:
            for head in self.prior_headers:
                mu_list.append(head(h))
            pi=None
        else:
            pi=0
            for head in self.prior_headers:#Assuming only 1, just cant be bothered to change it.
                mu_list.append(head[0](h))
                logvar= head[1](h)
                logvar_list.append(logvar)
                #mu = head[0](h) 
                #raw_chol = head[1](h)
                #L = self.chol_from_raw(raw_chol)
                #mu_list.append(mu)
                #logvar_list.append(L)
        return mu_list, logvar_list, pi

    
    



    @tf.function(jit_compile=False)
    def call(self, h, mask, epoch, training):
        mu_list, logvar_list, pi = self.encode(h, mask, training=training)

        if self.Baseline_mode:
            z_list=mu_list
            y_hat_list=[]
            sigma_list=[]
        else:
            z_list=[]
            y_hat_list=[]
            sigma_list=[]
            B = tf.shape(mu_list[0])[0]
            
            for mu, logvar in zip(mu_list, logvar_list):#Assumeing only 1, just cant be bothered to change it.
                sigma = tf.exp(0.5 * logvar) + 1e-5 
                #sigma = tf.matmul(logvar, logvar, transpose_b=True)
                sigma_list.append(sigma)
                for i in range(self.num_u_draw):
                    eps = tf.random.normal(tf.shape(mu))
                    z =mu + sigma * eps                       # live: phys_mahal/log_pi cancel -> no erratic 1/σ kick on mu
                    #print(z[0])
                    z_list.append(z)
                    const = tf.fill([B, 1], self.z_const)
                    z_3D = tf.concat([z, const], axis=-1)   # detach ONLY for the raytracer -> avoids Dr.Jit OOM

                    skip_decode=False
                    if skip_decode:
                        y_hat=0
                    else:
                        y_hat_c = self._py_decode(z_3D)
                       
                        y_hat = tf.reshape(y_hat_c, [B, self.num_anchors, -1, 2])
                       
                        if self.Use_FFT:
                            y_hat = dataloader.fft_time_axis(y_hat)

                        #y_hat = (y_hat - self.scaler_mean) / self.scaler_std
                        y_hat, _, _, _ = dataloader.add_noise_then_scale_dataset_batch(y_hat,
                                                                                0, tf.ones([B], dtype=tf.bool),
                                                                                0, self.scaler_mean, self.scaler_std,
                                                                                self.RT_snr_db, dataset_signal_power=self.prior_dict['signal_P'])
                      
                    y_hat_list.append(y_hat)

        return mu_list, logvar_list, sigma_list, z_list, y_hat_list, pi

        
    
    def decode(self, z):
        a, tau = rt.get_realization(self.ch, scenario=self.scenario, individual_sensor=z, run_in_batches=False)

        if self.atau_mode:
            a = tf.squeeze(a)
            a_pad, tau_pad = dataloader.pad_paths(a_list=a, tau_list=tau, P_fixed=128,
                                                a_dtype=a[0][0].dtype, tau_dtype=tau[0][0].dtype)
            h_all = tf.stack((tf.math.real(a_pad), tf.math.imag(a_pad), tau_pad), axis=-1)
            return h_all
        else:
            h_cal = self.bwcir.bandlimit_cir(self.time_axis, a, tau)
            if len(h_cal.shape) == 2:
                h_cal= tf.expand_dims(h_cal, axis=1)
            # Shape guard: degenerate sampled positions can make the raytracer collapse the
            # anchor axis, so h_cal comes back as the wrong shape (e.g. [B,1,T] -> later
            # reshaped into a bogus T). If it isn't [B, num_anchors, T], substitute a zero
            # CIR of the correct shape; the empty-power check below then turns it into the
            # same tiny noise as the NaN path, but with guaranteed dimensions.
            #T = len(self.time_axis)
            #B = tf.shape(z)[0]
            #n_a = int(h_cal.shape[-2]) if h_cal.shape[-2] is not None else self.num_anchors
            #n_t = int(h_cal.shape[-1]) if h_cal.shape[-1] is not None else T
            #if len(h_cal.shape) != 3 or n_a != self.num_anchors or n_t != T:
            #    tf.print("decode: h_cal had unexpected shape", tf.shape(h_cal),
            #             "expected [B,", self.num_anchors, ",", T, "] - using zero CIR")
            #    h_cal = tf.zeros([B, self.num_anchors, T], dtype=h_cal.dtype)
            real_part = tf.math.real(h_cal)
            imag_part = tf.math.imag(h_cal)
            is_nan = tf.math.is_nan(real_part) | tf.math.is_nan(imag_part)
            is_inf = tf.math.is_inf(real_part) | tf.math.is_inf(imag_part)
            is_bad = is_nan | is_inf 
            frac_bad = tf.reduce_mean(tf.cast(is_bad, tf.float32))
            if tf.reduce_any(is_bad):
                tf.print("Sionna returned bad values in h_cal. Fraction bad:", frac_bad)
            noise_real = tf.random.normal(tf.shape(h_cal), stddev=1e-4, dtype=tf.float32)
            noise_imag = tf.random.normal(tf.shape(h_cal), stddev=1e-4, dtype=tf.float32)
            noise_complex = tf.complex(noise_real, noise_imag)

            h_cal_safe = tf.where(is_bad, noise_complex, h_cal)

            power = tf.reduce_sum(tf.square(tf.abs(h_cal_safe)), axis=-1)          # [B*K, N_anchors]
            total_power = tf.reduce_mean(power, axis=-1)                           # [B*K]
            is_empty = total_power < 1e-12                                         # [B*K]
            is_empty = tf.reshape(is_empty, [-1])
            mask_empty = is_empty[:, None, None]                                   # broadcastable
            empty_noise_real = tf.random.normal(tf.shape(h_cal_safe), stddev=1e-4, dtype=tf.float32)
            empty_noise_imag = tf.random.normal(tf.shape(h_cal_safe), stddev=1e-4, dtype=tf.float32)
            empty_noise = tf.complex(empty_noise_real, empty_noise_imag)
            h_cal_final = tf.where(mask_empty, empty_noise, h_cal_safe)
            y_hat = tf.stack([tf.math.real(h_cal_final), tf.math.imag(h_cal_final)], axis=-1)

            return y_hat
    def _py_decode(self, z):
        def decode_py(z_np):
            z_tf = tf.convert_to_tensor(z_np, dtype=z.dtype)  # or tf.float32
            with tf.GradientTape(persistent=False) as _dummy_tape:  # dummy to force eager context if needed
                pass
            y_c = self.decode(z_tf)

            #y_c = tf.clip_by_value(y_c, -1e4, 1e4)  # reasonable range for CIR
            #y_c = tf.where(
            #    tf.math.is_nan(y_c) | tf.math.is_inf(y_c),
            #    tf.zeros_like(y_c, dtype=y_c.dtype),   # or tf.random.normal(..., stddev=1e-4)
            #    y_c
            #)
            #power = tf.reduce_sum(tf.square(tf.abs(y_c)), axis=-1, keepdims=True)
            #y_c = tf.where(power < 1e-12, tf.zeros_like(y_c), y_c)


            return y_c

        wrapped = tf.py_function(
            func=decode_py,
            inp=[z],
            Tout=tf.float32, 

            name="sionna_rt_bridge"
        )
        wrapped.set_shape([None, self.num_anchors, None, 2])
        # Key: stop gradients here (prevents backprop trying to go through unknown-shape output)
        return tf.stop_gradient(wrapped)
    
    
    def inference_call(self, h, CFR, mask,xyz_true,attempts=10, training=False, fingerprint_method="CONV"):
        sigma_list = []
        training = False
        mu_list, logvar_list, _ = self.encode(h, mask, training=training)
        CFR=tf.squeeze(CFR, axis=1)
        B = tf.shape(mu_list[0])[0]
        for mu, logvar in zip(mu_list, logvar_list):
            sigma = tf.exp(0.5 * logvar) + 1e-5
            #sigma = tf.matmul(logvar, logvar, transpose_b=True)
            sigma_list.append(sigma)
        # Assuming the last mu and sigma are the ones used for sampling (as per your code)
        mu = mu_list[-1]  # Use the last mu if multiple; adjust if needed
        sigma = sigma_list[-1]  # Use the last sigma
        attempts = attempts
        best_mse = tf.fill([B], tf.constant(float("inf"), tf.float32))
        best_y = tf.zeros([B], dtype=tf.int32)
        best_z = tf.zeros([B, tf.shape(mu)[-1] + 1], dtype=mu.dtype)
        
        # Collect all z samples here (list of tensors)
        all_z_list = []
        all_weights=[]
        for k in range(attempts):
            #eps = tf.random.normal([B, 2], dtype=mu.dtype)
            #z = mu + tf.einsum('bij,bj->bi', logvar, eps)
            eps = tf.random.normal(tf.shape(mu))
            z = mu + sigma * eps 
            #z = tf.random.truncated_normal(
            #    tf.shape(mu),
            #    mean=mu,
            #    stddev=sigma
            #)
            all_z_list.append(z)  # Collect z before concat
            
            const = tf.ones([B, 1], dtype=z.dtype) * 1.15
            z_with_const = tf.concat([z, const], axis=-1)  # [B, latent+1]
            y_hat = self.decode(z_with_const)

            # y_hat = tf.stack([tf.math.real(y_hat_c), tf.math.imag(y_hat_c)], axis=-1)  # Uncomment if needed
            y_hat = tf.reshape(y_hat, [B, self.num_anchors, -1, 2])
            if self.Use_FFT:
                y_hat = dataloader.fft_time_axis(y_hat)
            # print(y_hat.shape)
            #y_hat = (y_hat - self.scaler_mean) / self.scaler_std
            y_hat, _, _, _ = dataloader.add_noise_then_scale_dataset_batch(y_hat, 
                                                                    0, tf.ones([B], dtype=tf.bool),
                                                                    0, self.scaler_mean, self.scaler_std, 
                                                                    self.RT_snr_db, dataset_signal_power=self.prior_dict['signal_P'])
            if self.use_MAG_inference:
                y_hat=dataloader.cir_to_pdp2(y_hat)
            #self.ssim_2d_rect(y_hat, CFR)
            if fingerprint_method=="CONV":
                weight=self.compute_conv(y_hat, CFR)
                weight=tf.reduce_max(weight,axis=1)
            elif fingerprint_method=="SSIM":
                weight=self.ssim_dist_scalar(y_hat, CFR, win_n=min(self.num_anchors, 5), win_t=255)
            elif fingerprint_method=="MSE":
                weight=1/tf.reduce_mean((y_hat - CFR)**2, axis=[1,2,3])
            all_weights.append(1-weight)
            mse=1-weight#mse = tf.reduce_mean(1 - weight, axis=[1, 2, 3])
            better = mse < best_mse
            best_mse = tf.where(better, mse, best_mse)
            # best_y = tf.where(better, y_hat, best_y)
            better_bc = better[:, None]  # [B,1]

            best_z = tf.where(better_bc, z_with_const, best_z)
        # print(best_z)
        # exit()
        
        # Stack the collected z's and convert to numpy for saving
        all_z = tf.stack(all_z_list)
        #weights = tf.stack(all_weights)
        #exp_w = tf.exp(-(1/0.1)*weights)
        #weights = exp_w / tf.reduce_sum(exp_w, axis=0, keepdims=True)
        #weighted_z = tf.einsum('kb,kbj->bj', weights, all_z)
        mu_np = mu.numpy()
        xyz = xyz_true[..., :2].numpy()
        all_z_np = all_z.numpy()  # [200, B, latent]
        
        # Save to .pkl file
        with open('inference_clustering.pkl', 'wb') as f:
            pickle.dump({'mu': mu_np,'z':best_z,'sigma':sigma.numpy(),'xyz':xyz, 'guesses': all_z_np}, f)
        return best_z,mu, sigma#best_z, sigma
    
    def baseline_loss(self,mu, L_R, L, u_true, alpha, y_hat, y_obs, epoch):
        pos_err = tf.reduce_mean((mu - u_true[..., :2])**2)
        return pos_err, {"term1":pos_err, "term2": tf.constant(0.0,tf.float32),"term3":tf.constant(0.0,tf.float32), "term4": tf.constant(0.0,tf.float32)}
    
    @tf.function(jit_compile=False)
    def loss_A(self, mu_list, logvar_list, sigma_list, z_list, y_hat_list, pi,
                    u_true, y_obs, epoch, alpha=0.01, fingerprint_method="CONV"):
        mu_stacked     = tf.stack(mu_list, axis=1)          # [B, K, 2]
        sigma_stacked  = tf.stack(sigma_list, axis=1)       # [B, K, 2]
        z_2d_stacked   = tf.stack(z_list)           # [B, K, 2]   (sampled positions)
        pi_stacked     = pi                                 # [B, K]
        y_hat_stacked  = tf.stack(y_hat_list)       # [K, B, N_anchors, time, 2]
        L_stacked = tf.stack(logvar_list, axis=1)
        # ── Raytracer-based soft weights (unchanged logic, just adapted) ─────────
        ssim_per_comp=[]
        #if self.num_anchors!=1:
        y_obs=tf.squeeze(y_obs,axis=1)
        tau=0.01
        print("Tau:", tau)
        for i in range(self.num_u_draw):
            if self.use_MAG_inference:
                y_obs=dataloader.cir_to_pdp2(y_obs)
                y_hat_stacked[i,:]=dataloader.cir_to_pdp2(y_hat_stacked[i,:])
            if fingerprint_method=="CONV":
                #weight=self.compute_conv(y_hat_stacked[i,:], y_obs)
                #weight=tf.reduce_max(weight,axis=1)
                weight= tf.ones_like(u_true[..., 0], dtype=tf.float32)
            elif fingerprint_method=="SSIM":
                weight=self.ssim_dist_scalar(y_hat_stacked[i,:], y_obs, win_n=min(self.num_anchors, 5), win_t=127)
                
                #weight= tf.ones_like(u_true[..., 0], dtype=tf.float32)*0.5
            ssim_per_comp.append(1-weight)
        w_stacked = tf.stack(ssim_per_comp)
        exp_w = tf.exp(-(1.0/tau)*w_stacked)
        E = exp_w / tf.reduce_sum(exp_w, axis=0, keepdims=True)
        #E = -(1.0 / tau) * w_stacked         # raw energy, shape [K, B]
        best_idx = tf.argmax(E, axis=0)       # select highest-energy sample
        L     = tf.squeeze(tf.linalg.diag(sigma_stacked))
        mu_stacked=tf.squeeze(mu_stacked)

        pos_err = VAE_TF.position_mahalanobis_error(mu_stacked, L, u_true)


        phys_mahal_list=[]
        #z_error_list=[]
        for i in range(self.num_u_draw):
            phys_mahal_list.append(VAE_TF.position_mahalanobis_error(mu_stacked, L, z_2d_stacked[i,:]))
        #    z_error_list.append(VAE_TF.position_mahalanobis_error(u_true, L, z_2d_stacked[i,:]))
        phys_mahal = tf.stack(phys_mahal_list)

        log_pi = VAE_TF.log_normal_diag(
            z_2d_stacked,
            mu_stacked[None, ...],
            tf.squeeze(sigma_stacked[None, ...],axis=2)
        )   # [K, B]

        #batch_idx = tf.range(tf.shape(best_idx)[0])
        batch_idx = tf.range(tf.shape(best_idx)[0], dtype=best_idx.dtype)

        log_pi_pos = tf.gather_nd(
            log_pi,
            tf.stack([best_idx, batch_idx], axis=1)
        )

        E_pos = tf.gather_nd(
            E,
            tf.stack([best_idx, batch_idx], axis=1)
        )

        log_denom = tf.reduce_logsumexp(E, axis=0) - tf.math.log(
            tf.cast(self.num_u_draw, tf.float32)
        )

        #L_phys_new = -(log_pi_pos + E_pos - log_denom)
        cov_err = VAE_TF.log_det(L)#VAE_TF.log_det_cov_from_chol(L)
        lamb=1#tf.where(epoch < 200.0, 0.005 * epoch, 1)
        L_sup=(pos_err+0.5*cov_err)#+tf.reduce_mean((mu_stacked - u_true[..., :2])**2)
        #L_phys=lamb*phys_mahal*E+tf.einsum('b,kb->kb', cov_err, E)
        # σ log-det only (the -log_pi piece that survives), no quadratic, so detaching z leaves phys_mahal as the sole quadratic
        sigma_b      = tf.squeeze(sigma_stacked[None, ...], axis=2)          # [1, B, 2]
        logdet_sigma = 0.5 * tf.reduce_sum(tf.math.log(sigma_b**2), axis=-1) # [1, B]
        L_phys       = lamb*phys_mahal*E + logdet_sigma*E                    # [K, B]

        loss = tf.reduce_mean(L_sup)+tf.reduce_mean(tf.reduce_sum(L_phys,axis=0))#tf.reduce_mean(L_phys_new)#tf.reduce_mean(tf.reduce_sum(L_phys,axis=0))
        #loss= tf.reduce_mean(L_sup) + tf.reduce_mean(cov_err)
        best_z   = tf.gather(z_2d_stacked, best_idx, axis=0, batch_dims=1)
        best_y   = tf.gather(y_hat_stacked, best_idx, axis=0, batch_dims=1)
        
        extras = {
            "best_z":       best_z,
            "best_y_hat":   best_y,
            "sigma_mean":   tf.reduce_mean(sigma_stacked),
            "sigma": sigma_stacked,
            "weights":       w_stacked,
            "z": z_2d_stacked,
            "l_sup": L_sup,
            "l_phys": L_phys,
            "log_pi":log_pi_pos,
            "E_pos":E_pos,
            "log_denom":log_denom,
            "cov_err":cov_err
        }

        return loss, extras

    @tf.function(jit_compile=False)
    def loss_C(self, mu_list, logvar_list, sigma_list, z_list, y_hat_list, pi,
                    u_true, y_obs, epoch, alpha=0.01):
        
        mu_stacked     = tf.stack(mu_list, axis=1)          # [B, K, 2]
        sigma_stacked  = tf.stack(sigma_list, axis=1)       # [B, K, 2]
        z_2d_stacked   = tf.stack(z_list, axis=1)           # [B, K, 2]   (sampled positions)
        pi_stacked     = pi                                 # [B, K]
        y_hat_stacked  = tf.stack(y_hat_list, axis=1)       # [B, K, N_anchors, time, 2]

        phys_mahal_list=[]
        z_error_list=[]
        for i in range(self.num_u_draw):
            #phys_mahal_list.append(VAE_TF.position_mahalanobis_error(mu_stacked, L, z_2d_stacked[:,i]))
            #phys_mahal_list.append(VAE_TF.position_mahalanobis_error(u_true[:, :2], L, z_2d_stacked[:,i]))
            z_error_list.append(tf.reduce_mean(tf.square(z_2d_stacked[:,i] - u_true[:, :2]),axis=1))
        z_error_list= tf.stack(z_error_list, axis=1)

        lamb=1#tf.where(epoch < 200.0, 0.005 * epoch, 1)
       
        loss= tf.reduce_mean(z_error_list)

        extras = {
            "best_z":       z_error_list,
            "sigma_mean":   tf.reduce_mean(sigma_stacked),
            "z": z_2d_stacked,
        }

        return loss, extras
    
    @staticmethod
    def log_det(L):
        eps = 1e-6
        alpha=1
        L = tf.convert_to_tensor(L)
        dtype = L.dtype
        alpha = tf.cast(alpha, dtype)
        D = tf.shape(L)[-1]
        batch_shape = tf.shape(L)[:-2]
        I = tf.eye(D, batch_shape=batch_shape, dtype=dtype)
        A = tf.matmul(L, L, transpose_a=True)
        inv_alpha = 1.0 / (alpha + tf.cast(eps, dtype))
        M = I + inv_alpha * A
        C = tf.linalg.cholesky(M)
        logdet_M = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(C) + tf.cast(eps, dtype)), axis=-1)
        cov_err = 0.5*(tf.cast(D, dtype) * tf.math.log(alpha + tf.cast(eps, dtype)) + logdet_M)
        return cov_err
   
    @staticmethod
    def log_det_cov_from_chol(L, eps=1e-6):
        diag = tf.linalg.diag_part(L)                # [B, 2]
        diag = tf.maximum(diag, eps)
        return 2.0 * tf.reduce_sum(tf.math.log(diag), axis=-1)

    @staticmethod
    def position_mahalanobis_error(mu, L, u_true, eps=1e-6):
        mu_pos = mu[..., :2]       
        u_pos  = u_true[..., :2]     
        diff   = mu_pos - u_pos  
        L_pos = L[..., :2, :2]     
        Gamma = tf.matmul(L_pos, L_pos, transpose_b=True) 
        dim = tf.shape(Gamma)[-1]
        I = tf.eye(dim, dtype=Gamma.dtype)[None, :, :] 
        Gamma = Gamma + eps * I
        Gamma_inv = tf.linalg.pinv(Gamma) 
        mahal_vec = tf.einsum('bi,bij,bj->b', diff, Gamma_inv, diff) 
        pos_mahal = mahal_vec
        return pos_mahal
    
    @staticmethod
    def log_normal_diag(z, mu, sigma):
        return -0.5 * (
            tf.reduce_sum(((z - mu) / sigma)**2, axis=-1)
            + tf.reduce_sum(tf.math.log(sigma**2), axis=-1)
            + z.shape[-1] * tf.math.log(2.0 * tf.constant(np.pi))
        )

    
    
    @staticmethod
    def ssim_dist_scalar(y_hat, y_obs, win_n=3, win_t=11):
        dmap = VAE_TF.ssim_2d_rect_unsafe(y_hat, y_obs, win_n=win_n, win_t=win_t)
        #ssim = tf.debugging.check_numerics(dmap, "ssim contains inf/nan")
        #ssim = tf.clip_by_value(ssim, 0.0, 50.0)
        return tf.reduce_mean(dmap, axis=[1,2,3])
    @staticmethod
    def compute_conv(y_obs, y_hat):
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

    @staticmethod
    def gaussian_1d(size, sigma, dtype=tf.float32):
        r = tf.range(-(size//2), size//2 + 1, dtype=dtype)
        g = tf.exp(-(r*r) / (2.0 * sigma*sigma))
        return g / tf.reduce_sum(g)

    @staticmethod
    def ssim_2d_rect_unsafe(y_hat, y_obs, max_val=1.0,
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
        g_n = VAE_TF.gaussian_1d(win_n, sigma_n, dtype=x.dtype)  # [win_n]
        g_t = VAE_TF.gaussian_1d(win_t, sigma_t, dtype=x.dtype)  # [win_t]
        k2d = tf.tensordot(g_n, g_t, axes=0)              # [win_n, win_t]
        k2d = k2d / tf.reduce_sum(k2d)

        # Depthwise conv2d expects kernel [kh, kw, in_ch, channel_multiplier]++
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
        #sigma_xy = tf.clip_by_value(
        #    sigma_xy,
        #    -tf.sqrt(sigma_x2 * sigma_y2),
        #    tf.sqrt(sigma_x2 * sigma_y2)
        #)
        K1, K2 = 0.05, 0.15
        C1 = (K1 * max_val)**2
        C2 = (K2 * max_val)**2

        num = (2.0*mu_xy + C1) * (2.0*sigma_xy + C2)
        den_safe = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2) + 1e-10
        ssim_map = num / den_safe
        ssim_map = tf.where(tf.math.is_nan(ssim_map) | tf.math.is_inf(ssim_map), 0.0, ssim_map)
        return ssim_map
    @staticmethod
    def permute_tx(h, mask):
        # h: [B,N,T,2], mask: [B,N]
        N = tf.shape(h)[1]
        perm = tf.random.shuffle(tf.range(N))
        h_p = tf.gather(h, perm, axis=1)
        m_p = tf.gather(mask, perm, axis=1)
        return h_p, m_p
    
    @staticmethod
    def _to_pdp(y, eps=1e-8):
        y = tf.cast(y, tf.float32)
        re, im = y[..., 0], y[..., 1]
        return tf.maximum(re * re + im * im, eps)  # [B,N,T]
    

    def chol_from_raw(self, raw, eps=1e-4):
        # raw: [B, 3] = [a, b, c]
        a = raw[:, 0]
        b = raw[:, 1]
        c = raw[:, 2]

        l11 = tf.nn.softplus(a) + eps
        l21 = b
        l22 = tf.nn.softplus(c) + eps

        row1 = tf.stack([l11, tf.zeros_like(l11)], axis=-1)
        row2 = tf.stack([l21, l22], axis=-1)
        L = tf.stack([row1, row2], axis=1)   # [B, 2, 2]
        return L


