import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def per_anchor_cnn(embed_dim=128, weight_decay=1e-4, channels=2):
    """Encodes one anchor CIR of shape (L, channels) -> (embed_dim,)."""
    inp = keras.Input(shape=(None, channels))
    x = layers.Conv1D(32, 7 * 4, strides=2, padding="same", use_bias=False,
                      kernel_regularizer=keras.regularizers.l2(weight_decay))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool1D(channels)(x)

    x = layers.Conv1D(64, 5 * 4, strides=2, padding="same", use_bias=False,
                      kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool1D(channels)(x)

    x = layers.Conv1D(128, 3 * 4, padding="same", use_bias=False,
                      kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(embed_dim, activation="relu")(x)
    return keras.Model(inp, x, name="per_anchor_cnn")


class ManualMHA(layers.Layer):
    def __init__(self, embed_dim, num_heads, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        assert embed_dim % num_heads == 0
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = tf.constant(self.head_dim ** -0.5, dtype=tf.float32)
        self.q_proj = layers.Dense(self.embed_dim, use_bias=True)
        self.k_proj = layers.Dense(self.embed_dim, use_bias=True)
        self.v_proj = layers.Dense(self.embed_dim, use_bias=True)
        self.out_proj = layers.Dense(self.embed_dim, use_bias=True)

        self.attn_drop = layers.Dropout(dropout)

    def _split_heads(self, x):
        # x: (B, T, D) -> (B, H, T, Hd)
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]
        x = tf.reshape(x, [B, T, self.num_heads, self.head_dim])
        return tf.transpose(x, [0, 2, 1, 3])

    def _merge_heads(self, x):
        # x: (B, H, T, Hd) -> (B, T, D)
        x = tf.transpose(x, [0, 2, 1, 3])
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]
        return tf.reshape(x, [B, T, self.embed_dim])

    def call(self, x, attention_mask=None, training=False):
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        # Do attention math in fp32 for stability
        qf = tf.cast(q, tf.float32)
        kf = tf.cast(k, tf.float32)
        vf = tf.cast(v, tf.float32)

        scores = tf.matmul(qf, kf, transpose_b=True) * self.scale  # fp32

        if attention_mask is not None:
            m = attention_mask
            if m.shape.rank == 3:
                # (B,1,T) -> (B,1,1,T), (B,T,T) -> (B,1,T,T)
                m = m[:, :, tf.newaxis, :] if tf.shape(m)[1] == 1 else m[:, tf.newaxis, :, :]
            m = tf.cast(m, tf.bool)
            scores = tf.where(m, scores, tf.constant(-1e9, dtype=tf.float32))

        attn = tf.nn.softmax(scores, axis=-1)                 # fp32
        attn = self.attn_drop(attn, training=training)

        out = tf.matmul(attn, vf)                             # fp32
        out = self._merge_heads(out)                          # fp32
        out = self.out_proj(tf.cast(out, x.dtype))            # back to model dtype
        return out


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.attn = ManualMHA(embed_dim, num_heads, dropout=dropout, name="manual_mha")
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)

        self.ffn = keras.Sequential([
            layers.Dense(mlp_dim, activation="gelu"),
            layers.Dropout(dropout),
            layers.Dense(embed_dim, kernel_initializer="zeros", bias_initializer="zeros"),
        ])
        self.do = layers.Dropout(dropout)

    def call(self, x, attn_mask=None, training=False):
        y = self.attn(x, attention_mask=attn_mask, training=training)
        x = self.ln1(x + self.do(y, training=training))
        y = self.ffn(x, training=training)
        x = self.ln2(x + self.do(y, training=training))
        return x


class AttnPool(layers.Layer):
    """Learned attention pooling over tokens -> single vector."""
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.k = layers.Dense(embed_dim, use_bias=False)
        self.v = layers.Dense(embed_dim, use_bias=False)
        self.score = layers.Dense(1, use_bias=False)
        self.neg_inf_f32 = tf.constant(-1e9, dtype=tf.float32)

    def call(self, x, token_mask=None):
        # x: (B, N, D)
        s = self.score(tf.nn.tanh(self.k(x)))  # (B, N, 1)
        neg = tf.cast(self.neg_inf_f32, s.dtype)
        s = tf.where(token_mask[:, :, None], s, neg)
        w = tf.nn.softmax(s, axis=1)           # (B, N, 1)
        return tf.reduce_sum(w * self.v(x), axis=1)  # (B, D)


class AnchorTransformerEncoder(tf.keras.Model):
    """
    Input:
      h:    (B, N, L, C)
      mask: (B, N) bool (True = valid)
    Output:
      ctx:  (B, out_dim)
    """
    def __init__(
        self,
        N_anchors,
        cir_len,
        in_ch=2,
        embed_dim=256,
        num_heads=8,
        mlp_dim=512,
        num_blocks=4,
        dropout=0.1,
        weight_decay=1e-4,
        out_dim=256,
        name="anchor_transformer_encoder",
    ):
        super().__init__(name=name)
        self.embed_dim = embed_dim
        self.N = int(N_anchors)
        self.L = int(cir_len)
        self.C = int(in_ch)
        self.anchor_enc = per_anchor_cnn(embed_dim=embed_dim, weight_decay=weight_decay, channels=in_ch)

        self.id_emb = layers.Embedding(
            self.N, embed_dim,
            embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
        )
        self.anchor_ids = tf.constant(np.arange(self.N)[None, :], dtype=tf.int32)  # (1, N)

        self.drop_in = layers.Dropout(dropout)
        self.blocks = []
        for i in range(num_blocks):
            self.blocks.append(
                TransformerBlock(embed_dim, num_heads, mlp_dim, dropout, name=f"tx_block_{i}")
            )
        self.ln_out = layers.LayerNormalization(epsilon=1e-6)
        self.pool = AttnPool(embed_dim, name="attn_pool")

        self.out_mlp = keras.Sequential([
            layers.Dense(out_dim, activation="relu"),
            layers.Dropout(dropout),
        ], name="out_mlp")

    def call(self, h, mask, training=False):
        B = tf.shape(h)[0]
        x = tf.reshape(h, [B * self.N, self.L, self.C])
        tok = self.anchor_enc(x, training=training)
        D = tok.shape[-1]
        tok = tf.reshape(tok, [B, self.N, D])
        tok = tok + self.id_emb(self.anchor_ids)
        attn_mask = tf.logical_and(mask[:, None, :], mask[:, :, None])
        attn_mask = attn_mask[:, None, :, :]
        x = self.drop_in(tok, training=training)
        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask, training=training)
        x = self.ln_out(x)
        ctx = self.pool(x, token_mask=mask)
        ctx = self.out_mlp(ctx, training=training)
        return ctx


class WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, total_steps, warmup_frac=0.05, min_lr=1e-5):
        super().__init__()
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * warmup_frac)
        self.min_lr = min_lr
        self.pi = tf.constant(3.14159265)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        total_steps = tf.cast(self.total_steps, tf.float32)

        warmup_lr = self.base_lr * step / tf.maximum(1.0, warmup_steps)

        progress = (step - warmup_steps) / tf.maximum(1.0, total_steps - warmup_steps)
        progress = tf.clip_by_value(progress, 0.0, 1.0)
        cosine_lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
            1.0 + tf.cos(self.pi * progress)
        )

        return tf.where(step < warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "total_steps": self.total_steps,
            "warmup_frac": self.warmup_steps / self.total_steps,
            "min_lr": self.min_lr,
        }
