import os
import argparse
import random
import math
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

try:
    from tensorflow.keras.optimizers.experimental import AdamW as AdamWOptimizer
except Exception:
    try:
        from tensorflow.keras.optimizers import AdamW as AdamWOptimizer
    except Exception:
        AdamWOptimizer = None

def set_gpu_growth():
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass


def set_seed(seed):
    if seed is None:
        return
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_split(split_dir, img_size=(224, 224)):
    classes = [("benign", 0), ("malignant", 1)]
    images = []
    labels = []
    for cls_name, cls_idx in classes:
        cls_dir = os.path.join(split_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            path = os.path.join(cls_dir, fname)
            if not os.path.isfile(path):
                continue
            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                continue
            img = img.resize(img_size)
            images.append(np.array(img, dtype=np.float32))
            labels.append(cls_idx)
    x = np.array(images, dtype=np.float32)
    y = to_categorical(np.array(labels, dtype=np.int32), 2)
    return x, y


class ClassToken(layers.Layer):
    def build(self, input_shape):
        self.cls = self.add_weight(
            name="cls_token",
            shape=(1, 1, input_shape[-1]),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        batch = tf.shape(x)[0]
        cls = tf.broadcast_to(self.cls, [batch, 1, tf.shape(x)[-1]])
        return tf.concat([cls, x], axis=1)


class AddPositionEmbedding(layers.Layer):
    def __init__(self, num_patches, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.pos = self.add_weight(
            name="pos_embed",
            shape=(1, self.num_patches + 1, self.embed_dim),
            initializer="random_normal",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        return x + self.pos


def mlp_block(x, hidden_dim, out_dim, dropout):
    x = layers.Dense(hidden_dim, activation=tf.nn.gelu)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(out_dim)(x)
    x = layers.Dropout(dropout)(x)
    return x


def transformer_block(x, num_heads, key_dim, mlp_ratio, dropout):
    # Self-attention
    x_norm = layers.LayerNormalization(epsilon=1e-6)(x)
    attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout)(x_norm, x_norm)
    x = layers.Add()([x, attn])
    # MLP
    x_norm = layers.LayerNormalization(epsilon=1e-6)(x)
    embed_dim = int(x.shape[-1])
    mlp_hidden = embed_dim * mlp_ratio
    mlp = mlp_block(x_norm, mlp_hidden, embed_dim, dropout)
    x = layers.Add()([x, mlp])
    return x


def build_vit(
    img_size=224,
    patch_size=16,
    embed_dim=256,
    depth=6,
    num_heads=8,
    mlp_ratio=4,
    dropout=0.1,
    num_classes=2,
):
    if img_size % patch_size != 0:
        raise ValueError("img_size must be divisible by patch_size")
    if embed_dim % num_heads != 0:
        raise ValueError("embed_dim must be divisible by num_heads")

    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = layers.Rescaling(1.0 / 255.0)(inputs)

    # Patch embedding
    x = layers.Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size, padding="valid")(x)
    num_patches = (img_size // patch_size) ** 2
    x = layers.Reshape((num_patches, embed_dim))(x)

    # Add class token + position
    x = ClassToken()(x)
    x = AddPositionEmbedding(num_patches=num_patches, embed_dim=embed_dim)(x)
    x = layers.Dropout(dropout)(x)

    # Transformer encoder
    key_dim = embed_dim // num_heads
    for _ in range(depth):
        x = transformer_block(x, num_heads=num_heads, key_dim=key_dim, mlp_ratio=mlp_ratio, dropout=dropout)

    x = layers.LayerNormalization(epsilon=1e-6)(x)
    cls_token = x[:, 0]
    outputs = layers.Dense(num_classes, activation="softmax")(cls_token)
    return models.Model(inputs=inputs, outputs=outputs, name="ViT_baseline")


class WarmUpCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, total_steps, warmup_steps, min_lr=1e-6):
        super().__init__()
        self.base_lr = base_lr
        self.total_steps = max(1, int(total_steps))
        self.warmup_steps = max(1, int(warmup_steps))
        self.min_lr = min_lr

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        total_steps = tf.cast(self.total_steps, tf.float32)

        warmup_lr = self.base_lr * step / warmup_steps
        progress = (step - warmup_steps) / tf.maximum(1.0, total_steps - warmup_steps)
        cosine_lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + tf.cos(math.pi * progress))
        return tf.where(step < warmup_steps, warmup_lr, cosine_lr)


def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
        y = y_true[:, 1]
        p = y_pred[:, 1]
        pt = y * p + (1.0 - y) * (1.0 - p)
        alpha_t = y * alpha + (1.0 - y) * (1.0 - alpha)
        return -alpha_t * tf.pow(1.0 - pt, gamma) * tf.math.log(pt)
    return loss


def compute_metrics(y_true, y_pred):
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def append_metrics_csv(path, variant, seed, metrics):
    if not path:
        return
    header = "variant,seed,accuracy,precision,recall,specificity,f1,tp,tn,fp,fn"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            first = f.readline().strip()
        if first and first != header:
            raise ValueError(f"Metrics CSV header mismatch. Use a new file: {path}")
    line = ",".join(
        [
            variant,
            str(seed),
            f"{metrics['accuracy']:.6f}",
            f"{metrics['precision']:.6f}",
            f"{metrics['recall']:.6f}",
            f"{metrics['specificity']:.6f}",
            f"{metrics['f1']:.6f}",
            str(metrics["tp"]),
            str(metrics["tn"]),
            str(metrics["fp"]),
            str(metrics["fn"]),
        ]
    )
    write_header = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if write_header:
            f.write(header + "\n")
        f.write(line + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", default="../data/train")
    parser.add_argument("--test_dir", default="../data/test")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--no_aug", action="store_true")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--mlp_ratio", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weights_out", default="vit_baseline.weights.h5")
    parser.add_argument("--weights", default="")
    parser.add_argument("--prefix", default="vit_baseline")
    parser.add_argument("--metrics_csv", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--use_warmup_cosine", action="store_true")
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--use_class_weight", action="store_true")
    parser.add_argument("--focal_loss", action="store_true")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--focal_alpha", type=float, default=0.25)
    args = parser.parse_args()

    set_gpu_growth()
    set_seed(args.seed)

    x_train, y_train = load_split(args.train_dir, img_size=(args.img_size, args.img_size))
    x_test, y_test = load_split(args.test_dir, img_size=(args.img_size, args.img_size))

    model = build_vit(
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        num_classes=2,
    )

    if args.weights:
        model.load_weights(args.weights)

    if AdamWOptimizer is None:
        raise RuntimeError(
            "AdamW optimizer is not available in this TensorFlow build. "
            "Please upgrade TensorFlow or use a build that provides AdamW."
        )

    steps_per_epoch = int(np.ceil(x_train.shape[0] / float(args.batch_size))) if x_train.size else 1
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch

    if args.use_warmup_cosine:
        lr_schedule = WarmUpCosine(args.lr, total_steps=total_steps, warmup_steps=warmup_steps, min_lr=args.min_lr)
        optimizer = AdamWOptimizer(learning_rate=lr_schedule, weight_decay=args.weight_decay)
    else:
        optimizer = AdamWOptimizer(learning_rate=args.lr, weight_decay=args.weight_decay)

    if args.focal_loss:
        loss_fn = focal_loss(gamma=args.focal_gamma, alpha=args.focal_alpha)
    else:
        loss_fn = "categorical_crossentropy"

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])

    if not args.eval_only:
        class_weight = None
        if args.use_class_weight and not args.focal_loss:
            y_labels = np.argmax(y_train, axis=1)
            counts = np.bincount(y_labels, minlength=2).astype(np.float32)
            total = float(np.sum(counts))
            class_weight = {i: total / (2.0 * counts[i]) for i in range(2)}
        if args.no_aug:
            history = model.fit(
                x_train,
                y_train,
                batch_size=args.batch_size,
                epochs=args.epochs,
                validation_data=(x_test, y_test),
                shuffle=True,
                class_weight=class_weight,
            )
        else:
            datagen = ImageDataGenerator(
                featurewise_center=False,
                samplewise_center=False,
                featurewise_std_normalization=False,
                samplewise_std_normalization=False,
                zca_whitening=False,
                rotation_range=0,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True,
                vertical_flip=False,
            )
            datagen.fit(x_train)
            history = model.fit(
                datagen.flow(x_train, y_train, batch_size=args.batch_size),
                steps_per_epoch=int(np.ceil(x_train.shape[0] / float(args.batch_size))),
                validation_data=(x_test, y_test),
                epochs=args.epochs,
                class_weight=class_weight,
            )

        train_acc = np.array(history.history.get("accuracy", []))
        test_acc = np.array(history.history.get("val_accuracy", []))
        train_loss = np.array(history.history.get("loss", []))
        test_loss = np.array(history.history.get("val_loss", []))

        np.savetxt(f"{args.prefix}-whole-train_acc.txt", train_acc)
        np.savetxt(f"{args.prefix}-whole-test_acc.txt", test_acc)
        np.savetxt(f"{args.prefix}-whole-train_loss.txt", train_loss)
        np.savetxt(f"{args.prefix}-whole-test_loss.txt", test_loss)

        model.save_weights(args.weights_out)

    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(model.predict(x_test, batch_size=args.batch_size), axis=1)
    metrics = compute_metrics(y_true, y_pred)
    append_metrics_csv(args.metrics_csv, "vit", args.seed, metrics)

    print("Eval metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
