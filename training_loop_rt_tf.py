import os
import numpy as np
import warnings
from collections import deque
# Silence Python warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import csv
import gc
import datetime
plt.rcParams['text.usetex'] = True
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_USE_LEGACY_KERAS"] = "0"
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.config.set_soft_device_placement(True)
physical_devices = tf.config.list_physical_devices('GPU')

for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
#Needs to be set right at the original TF import
seed = 42
tf.random.set_seed(seed)
tf.keras.utils.set_random_seed(seed)
np.random.seed(seed)
gpu_num = ""
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
#tf.config.set_soft_device_placement(False)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from VAE_RT import VAE_TF
import datahandling
import time
from transformer_model import WarmupCosine
print(tf.config.list_physical_devices("GPU"))
class trainer:
    def __init__(self, dataset=None, save_model_location=None, input=None,
             load_model=False, batch_size=128, train_snr_db=40.0,
             val_snr_db=30.0, RT_snr_db=0,
               sub_batch=None, Baseline_mode=False, val_mode="All"):
        print("Running:", dataset)
        print("Saving at: ", save_model_location)
        self.model_folder="models/"
        model_prefix="best_model"
        self.output_file="loss_terms.csv"
        self.fieldnames = ['type','epoch', 'term1', 'term2', 'term3', 'term4']
        self.npz_path=dataset
        self.epochs     = 500
        self.alpha      = 0.01#0.01
        self.use_FFT = False   #Only affects fingerprinting, encoder always gets non-FFT
        self.fingerprinting_method="SSIM" #"SSIM"|"CONV"
        self.use_MAG_encoder= False
        self.use_MAG_fingerprint = False
        self.a_tau_mode=False
        model_type='Transformer' #"Transformer"|"CNN"
        self.Baseline_mode=Baseline_mode
        self.learning_rate=5e-4

        self.dataloader=datahandling.dataloader(self.npz_path, train_split=0.8, a_tau_mode=self.a_tau_mode)
        if sub_batch is not None:
            self.low_memory=True
            self.batch_size=sub_batch
            self.batch_size_tar =   sub_batch*int(batch_size/sub_batch)#256
            self.batches_before_backpropagate = max(self.batch_size_tar, self.batch_size) // self.batch_size
        else:   
            self.low_memory=False
            self.batch_size=batch_size
        print("Using batch size", self.batch_size)

        self.train_ds, self.test_ds, self.val_ds, tx_all, self.Dy, self.mean_h, self.std_h, self.channels, self.prior_dict = \
        self.dataloader.make_dataset_shuffled(
            batch_size=self.batch_size,
            train_snr_db=train_snr_db,
            val_snr_db=val_snr_db,
            FFT=self.use_FFT
        )
        if val_mode != "All":
            self.val_ds = self.dataloader.rebuild_val_ds(
                val_mode, "datasets/LoS_mask.csv", self.batch_size, val_snr_db
            )

        self.mini_bs=len(tx_all)
        self.train_step=tf.function(self._train_step)
        self.val_step=tf.function(self._val_step)
        latent_dim = 2
        model_path, self.current_num, self.future_path=self.dataloader.find_latest(self.model_folder,model_prefix)
        
        if save_model_location is not None:
            self.future_path=save_model_location
        self.model = VAE_TF(self.batch_size, self.mini_bs, input_dim=self.Dy, 
                            latent_dim=latent_dim, channels=self.channels,
                            use_fft=self.use_FFT, RT_snr_db=RT_snr_db,
                            scaler_mean=self.mean_h, scaler_std=self.std_h, tx=tx_all,
                            prior_dict=self.prior_dict,
                            a_tau_mode=self.a_tau_mode,
                            Baseline_mode=self.Baseline_mode, MAG_inference=self.use_MAG_fingerprint,
                            model_type=model_type)
        
        dummy_h = tf.zeros(
            (self.batch_size, self.mini_bs, self.Dy, self.channels),
            dtype=tf.float32,        )
        _ = self.model(dummy_h, tf.ones((self.batch_size, self.mini_bs), dtype=tf.bool), epoch=0,training=False)
        if load_model:
            if ".keras" in save_model_location:
                model_path=save_model_location
            else:
            #if not ".keras" in save_model_location:
                for i in range(100):
                    test_path=save_model_location+str(i)+".keras"
                    if os.path.exists(test_path):
                        model_path=test_path
                    else:
                        break
            if os.path.exists(model_path):
                print("loading: ", model_path)
                self.model.load_weights(model_path)
            else:
                print("No pretrained weights found, starting from scratch.")

        self.accum_count = tf.Variable(0, dtype=tf.int32, trainable=False, name="accum_count")
        self.accum_grads = [
            tf.Variable(tf.zeros_like(v), trainable=False, name=f"accum_{i}")
            for i, v in enumerate(self.model.trainable_variables)
        ]
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.train_writer = tf.summary.create_file_writer(log_dir + "/train")

        with open("positions.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["mu_x", "mu_y", "true_x", "true_y","pos_err", "trace"])  # header

        self.total_steps = self.epochs *self.dataloader.dataset_len//self.batch_size
        lr_schedule = WarmupCosine(base_lr=self.learning_rate,total_steps=self.total_steps,warmup_frac=0.05,min_lr=5e-6,)# for 5/8 token
        self.opt = tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=1e-5,
            beta_1=0.9,
            beta_2=0.995,
            epsilon=1e-8,
            clipnorm=2.0)



    def position_error(self,mu_post, xyz_true, uncertainty, write_csv=False):
        mu_post  = tf.reshape(mu_post[..., :2], [-1, 2])
        xyz_true = tf.reshape(xyz_true[..., :2], [-1, 2])
        diff = mu_post - xyz_true
        return tf.sqrt(tf.reduce_mean(tf.reduce_sum(diff**2, axis=1))), diff

    def grad_accumulater(self, accumulated_grads, grads):
        new_accum = []
        for ag, g in zip(accumulated_grads, grads):
            if g is None:
                new_accum.append(ag)
            else:
                new_accum.append(ag + g)
        return new_accum
    
    
    
    @tf.function(jit_compile=False)
    def _train_step(self, h_b, xyz_b, mask, epoch, low_memory=False):
        
        with tf.GradientTape() as tape:
            if self.use_FFT:
                h_in=self.dataloader.unscale(h_b, self.dataloader.CFR_mean, self.dataloader.CFR_std)
                h_in=datahandling.dataloader.ifft_time_axis(h_in)
                h_in=self.dataloader.scale(h_in, self.dataloader.CIR_mean, self.dataloader.CIR_std)
            else:
                h_in=h_b
            if self.use_MAG_encoder:
                h_in=self.dataloader.cir_to_pdp2(h_in)
                
            mu, log_Var, sigma, z, y_hat, pi= self.model(h_in, mask, epoch=epoch, training=True)

            if self.use_MAG_fingerprint and not(self.Baseline_mode):
                y_hat=self.dataloader.cir_to_pdp2(y_hat)
                h_b = self.dataloader.cir_to_pdp2(h_b)

            if self.Baseline_mode:
                loss, _ = self.model.baseline_loss(mu, None, None, xyz_b, self.alpha, y_hat, y_obs=h_b, epoch=epoch)
            else:
                
                loss, _ = self.model.loss_A(mu, log_Var, sigma, z, y_hat, pi,
                        xyz_b[..., :2], h_b, epoch, fingerprint_method=self.fingerprinting_method)

        grads = tape.gradient(loss, self.model.trainable_variables)

        if not low_memory:
            self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
            return loss

        for ag, g in zip(self.accum_grads, grads):
            if g is not None:
                ag.assign_add(g)
        self.accum_count.assign_add(1)

        K = tf.cast(self.batches_before_backpropagate, tf.int32)
        def apply_and_reset():
            denom = tf.cast(self.accum_count, self.accum_grads[0].dtype)
            self.opt.apply_gradients(zip([ag / denom for ag in self.accum_grads], self.model.trainable_variables))
            for ag in self.accum_grads:
                ag.assign(tf.zeros_like(ag))
            self.accum_count.assign(0)
            return 0

        tf.cond(self.accum_count >= K, apply_and_reset, lambda: 0)
        return loss
    

    def train_step_debug(self,h_b, xyz_b, mask):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(h_b)
            mu, log_Var, sigma, z, y_hat, pi= self.model(h_b, mask, epoch=0, training=True)
            _, losses = self.model.loss_A(mu, log_Var, sigma, z, y_hat, pi,
                        xyz_b[..., :2], h_b, 0)
            loss=losses["log_pi"]    

        grads = tape.gradient(loss, mu)
        print("Mean absolute gradient:", tf.reduce_mean(tf.abs(grads)))
        return loss
    def tensor_dict_to_py(self, d):
        out = {}
        for k, v in d.items():
            if tf.is_tensor(v):
                # Ensure scalar
                v = v.numpy()
            # Handle numpy scalars cleanly
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                out[k] = str(v)
        return out
    @tf.function(jit_compile=False)
    def _val_step(self, h_b, xyz_b, mask, epoch, inference=False):
        write_csv=inference
        if self.use_FFT:
            h_in=self.dataloader.unscale(h_b, self.dataloader.CFR_mean, self.dataloader.CFR_std)
            h_in=datahandling.dataloader.ifft_time_axis(h_in)
            h_in=self.dataloader.scale(h_in, self.dataloader.CIR_mean, self.dataloader.CIR_std)
        else:
            h_in=h_b
        if self.use_MAG_encoder:
            h_in=self.dataloader.cir_to_pdp2(h_in)
        if self.use_MAG_fingerprint:
            h_b=self.dataloader.cir_to_pdp2(h_b)
        if self.Baseline_mode:
            mu, log_Var, sigma, z, y_hat, pi= self.model(h_in, mask, epoch=epoch, training=False)
            loss, _ = self.model.baseline_loss(mu, None, None, xyz_b, self.alpha, y_hat, y_obs=h_b, epoch=epoch)
            pe, diff = self.position_error(tf.stack(mu, axis=1), xyz_b, tf.stack(mu, axis=1)*0, write_csv=write_csv)
            sigma= tf.ones_like(mu, dtype=tf.float32)
            if inference:
                import pickle
                with open('inference_clustering.pkl', 'wb') as f:
                    pickle.dump({'mu': tf.squeeze(mu).numpy(),'z':tf.squeeze(mu).numpy(),'sigma':sigma.numpy(),'xyz':xyz_b.numpy()}, f)
        elif inference:
            z, mu, sigma= self.model.inference_call(h_in,h_b, mask,xyz_b,attempts=200, training=False,fingerprint_method=self.fingerprinting_method)
            pe, diff = self.position_error(z, xyz_b, sigma, write_csv=write_csv)
            loss=tf.constant(0.0)
        else:
            mu, log_Var, sigma, z, y_hat, pi= self.model(h_in, mask, epoch=epoch, training=False)
            loss, extras = self.model.loss_A(mu, log_Var, sigma, z, y_hat, pi,
                    xyz_b[..., :2], h_b, epoch,fingerprint_method=self.fingerprinting_method)
            pe, diff = self.position_error(tf.stack(mu, axis=1), xyz_b, extras['sigma_mean'], write_csv=write_csv)
            with self.train_writer.as_default():
                l_sup_mean    = tf.reduce_mean(extras['l_sup'])      # or tf.reduce_sum(...) / tf.cast(tf.size(...), tf.float32)
                l_phys_mean   = tf.reduce_mean(extras['l_phys'])     # assuming you have/fix this one
                cov_err    = tf.reduce_mean(extras['cov_err'])

                tf.summary.scalar("val/L_sup",     l_sup_mean,    step=tf.cast(epoch, tf.int64))
                tf.summary.scalar("val/L_phys",    l_phys_mean,   step=tf.cast(epoch, tf.int64))
                tf.summary.scalar("val/cov_err", cov_err,   step=tf.cast(epoch, tf.int64))
                tf.summary.scalar("val/RMSE", pe,   step=tf.cast(epoch, tf.int64))
            #loss, _ = self.model.simplified_loss(mu,z, L_R, L, xyz_b, self.alpha, y_hat, y_obs=h_b, epoch=epoch)
            sigma=extras['sigma_mean']
        return loss, pe, diff, mu, sigma

    def training_loop(self):
        best_val = 90
        pos_err = deque(maxlen=10)
        for epoch in range(1, self.epochs + 1):
            epoch_t = tf.convert_to_tensor(epoch, dtype=tf.float32)
            tr_metric = tf.keras.metrics.Mean()
            for h_b, xyz_b, mask, id_b in self.train_ds:
                
                loss= self.train_step(h_b, xyz_b, mask, epoch_t, low_memory=self.low_memory)
                #loss = self.train_step_debug(h_b, xyz_b, mask)
                tr_metric.update_state(loss)
            tr_avg = tr_metric.result().numpy()
            if epoch%1==0:
                
                va_total = tf.zeros([], tf.float32)
                va_steps = tf.zeros([], tf.float32)

                pe_total = tf.zeros([], tf.float32)
                pe_count = tf.zeros([], tf.float32)
                
                for h_b, xyz_b, mask, id_b in self.test_ds:
                    loss, pe, diff, _, _ = self.val_step(h_b, xyz_b, mask, epoch_t)
                    va_total += tf.cast(loss, tf.float32)
                    va_steps += 1.0

                    # pe: could be scalar or vector
                    pe = tf.cast(pe, tf.float32)
                    pe_total += tf.reduce_sum(pe)
                    pe_count += tf.cast(tf.size(pe), tf.float32)

                va_avg = (va_total / tf.maximum(1.0, va_steps)).numpy()
                pos_avg = (pe_total / tf.maximum(1.0, pe_count)).numpy()
                pos_err.append(pos_avg)
                #print("Pos error:", pos_avg)
                #print("val_error:",va_avg)
                if np.mean(pos_err) < best_val and epoch>10:
                    best_val = np.mean(pos_err)
                    self.model.save(self.future_path+str(int(epoch*0.02))+'.keras')
                print(f"Epoch {epoch:03d} | train {tr_avg:.4f} | val {va_avg:.4f} |pos {pos_avg:.1f} | best {best_val:.4f}")
            #if epoch ==50 or epoch ==100:
            #    self.opt.learning_rate.assign(self.opt.learning_rate * 0.5)
        return best_val
        

    def validation_loop(self):
        va_total = 0.0
        va_steps = 0
        mu_list=[]
        xyz_list=[]
        uncertainty_list=[]
        pe_list=[]
        diff_list=[]
        for h_b, xyz_b, mask, id_b in self.val_ds:
            loss, pe, diff, mu, sigma = self.val_step(h_b, xyz_b, mask, 0, inference=True)
            pe_list.append(pe.numpy())
            diff_list.extend(diff.numpy())
            mu_list.extend(mu)
            uncertainty_list.extend(sigma)
            xyz_list.extend(xyz_b)
            va_total += float(loss)
            va_steps += 1
        dist = np.linalg.norm(np.array(diff_list), axis=1) 
        median_dist = np.median(dist) 
        mean_dist=np.mean(dist)
        va_avg = va_total / max(1, va_steps)
        print(f"Validation loss: {va_avg:.4f}")
        pos_avg = sum(pe_list) / len(pe_list)
        print("RMSE: ", pos_avg, "MAE: ", mean_dist, "Median: ", median_dist)
        with open("positions.csv", "a", newline="") as f:
                writer = csv.writer(f)
                for i in range(np.shape(mu_list)[0]):
                    writer.writerow([mu_list[i][0].numpy(), mu_list[i][1].numpy(), xyz_list[i][0].numpy(), xyz_list[i][1].numpy(),tf.reduce_mean(dist[i]).numpy(), uncertainty_list[i].numpy()])

if __name__ == "__main__":
    datasets=["datasets/6_munich_5AP.pkl","datasets/6_munich_7AP.pkl", "datasets/6_munich_10AP.pkl", "datasets/6_munich_12AP.pkl", "datasets/6_munich_15AP.pkl"]
    save_names=["overnight/mAP5","overnight/mAP7", "overnight/large_mAP10", "overnight/large_mAP12","overnight/mAP15"]
    pre_trained_models=["models/26051500.keras","models/27051600.keras","models/25051000.keras","models/27050000.keras","models/8060600.keras"]
    Validate_model=False
    tf.config.run_functions_eagerly(Validate_model)
    batch_size=512
    #val_data_noise=0.05
    #If you want to save VRAM/RAM by splitting into smaller batches, that only backpropagate when 
    #summed up to batch_size, set small_batch to wished for sub batch size other than None.
    small_batch=None
    Baseline_mode=False
    train_snr_db = 25.0
    val_snr_db = 25.0
    RT_snr_db = 4000.0#4000
    print("train/val/RT SNR: ", train_snr_db,  '/' , val_snr_db, '/' , RT_snr_db)
    trainer_instance = trainer(
        datasets[0],
        save_model_location=pre_trained_models[0],
        input=None,
        load_model=Validate_model,
        batch_size=batch_size,
        train_snr_db=train_snr_db,
        val_snr_db=val_snr_db,
        RT_snr_db=RT_snr_db,
        sub_batch=small_batch,
        Baseline_mode=Baseline_mode,
        val_mode="All"
    )

    if Validate_model:
        trainer_instance.validation_loop()
    else:
        trainer_instance.training_loop()
    exit()
    train=False
    if train:
        with open("Overnight_eval.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["dataset", "test_pos_error"])
    else:

        tf.config.run_functions_eagerly(True)
    for data, path in zip(datasets, save_names):
        tf.keras.backend.clear_session()
        gc.collect()

        trainer_instance = trainer(data, path, input=None, load_model=True)
        if train:
            best_val=trainer_instance.training_loop()
            with open("Overnight_eval.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([data, best_val])
        else:
            trainer_instance.validation_loop()


        
        
        

        del trainer_instance
        tf.keras.backend.clear_session()
        gc.collect()
    #trainer_instance.validation_loop_save_npz()
