import os
import csv
import argparse
import random
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator


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
                img = Image.open(path).convert('RGB')
            except Exception:
                continue
            img = img.resize(img_size)
            images.append(np.array(img, dtype=np.float32))
            labels.append(cls_idx)
    x = np.array(images, dtype=np.float32)
    y = to_categorical(np.array(labels, dtype=np.int32), 2)
    x = preprocess_input(x)
    return x, y


def _open_metadata(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";"])
        except Exception:
            dialect = csv.get_dialect("excel")
        reader = csv.DictReader(f, dialect=dialect)
        rows = list(reader)
    return reader.fieldnames or [], rows


def load_split_from_metadata(
    metadata_path,
    image_root,
    split_value,
    img_size=(224, 224),
    image_col="isic_id",
    label_col="diagnosis_1",
    split_col="split",
    label_pos="malignant",
    label_neg="benign",
    image_ext=".jpg",
):
    fieldnames, rows = _open_metadata(metadata_path)
    if image_col not in fieldnames:
        raise ValueError(f"image_col '{image_col}' not found in metadata. Available: {fieldnames}")
    if label_col not in fieldnames:
        raise ValueError(f"label_col '{label_col}' not found in metadata. Available: {fieldnames}")
    if split_col and split_col not in fieldnames:
        raise ValueError(f"split_col '{split_col}' not found in metadata. Available: {fieldnames}")

    pos_set = {label_pos.lower()}
    neg_set = {label_neg.lower()}

    images = []
    labels = []
    for row in rows:
        if split_col and split_value is not None:
            if row.get(split_col, "").strip().lower() != str(split_value).strip().lower():
                continue
        label_raw = row.get(label_col, "").strip().lower()
        if label_raw in pos_set:
            label = 1
        elif label_raw in neg_set:
            label = 0
        else:
            continue

        name = row.get(image_col, "").strip()
        if not name:
            continue
        if not os.path.splitext(name)[1]:
            name = name + image_ext

        if os.path.isabs(name):
            path = name
        else:
            base = image_root or ""
            path = os.path.join(base, name)
            if not os.path.isfile(path):
                subdir = "malignant" if label == 1 else "benign"
                alt = os.path.join(base, subdir, name)
                if os.path.isfile(alt):
                    path = alt
                else:
                    continue
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            continue
        img = img.resize(img_size)
        images.append(np.array(img, dtype=np.float32))
        labels.append(label)

    x = np.array(images, dtype=np.float32)
    y = to_categorical(np.array(labels, dtype=np.int32), 2)
    x = preprocess_input(x)
    return x, y


def load_data(
    split_dir,
    img_size,
    metadata_path="",
    image_root="",
    split_value=None,
    image_col="isic_id",
    label_col="diagnosis_1",
    split_col="split",
    label_pos="malignant",
    label_neg="benign",
    image_ext=".jpg",
):
    if metadata_path:
        return load_split_from_metadata(
            metadata_path,
            image_root or split_dir,
            split_value,
            img_size=img_size,
            image_col=image_col,
            label_col=label_col,
            split_col=split_col,
            label_pos=label_pos,
            label_neg=label_neg,
            image_ext=image_ext,
        )
    return load_split(split_dir, img_size=img_size)


def build_mobilenet(input_shape=(224, 224, 3), num_classes=2, weights="imagenet"):
    inputs = layers.Input(shape=input_shape)
    base = MobileNetV2(weights=weights, include_top=False, input_tensor=inputs)
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(2048, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs=inputs, outputs=outputs, name='MobileNetV2')


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
    parser.add_argument('--train_dir', default='./train')
    parser.add_argument('--test_dir', default='./test')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--no_aug', action='store_true')
    parser.add_argument('--weights_out', default='mobilenet_path_to_my_weights.weights.h5')
    parser.add_argument('--weights', default='')
    parser.add_argument('--prefix', default='mobilenet')
    parser.add_argument('--metrics_csv', default='')
    parser.add_argument('--variant', default='mobilenet')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--no_pretrained', action='store_true')
    parser.add_argument('--metadata', default='')
    parser.add_argument('--image_root', default='')
    parser.add_argument('--image_col', default='isic_id')
    parser.add_argument('--label_col', default='diagnosis_1')
    parser.add_argument('--split_col', default='split')
    parser.add_argument('--train_split', default='train')
    parser.add_argument('--test_split', default='test')
    parser.add_argument('--label_pos', default='malignant')
    parser.add_argument('--label_neg', default='benign')
    parser.add_argument('--image_ext', default='.jpg')
    args = parser.parse_args()

    set_gpu_growth()
    set_seed(args.seed)

    if args.eval_only:
        x_train, y_train = np.array([]), np.array([])
        x_test, y_test = load_data(
            args.test_dir,
            img_size=(224, 224),
            metadata_path=args.metadata,
            image_root=args.image_root,
            split_value=args.test_split,
            image_col=args.image_col,
            label_col=args.label_col,
            split_col=args.split_col,
            label_pos=args.label_pos,
            label_neg=args.label_neg,
            image_ext=args.image_ext,
        )
    else:
        x_train, y_train = load_data(
            args.train_dir,
            img_size=(224, 224),
            metadata_path=args.metadata,
            image_root=args.image_root,
            split_value=args.train_split,
            image_col=args.image_col,
            label_col=args.label_col,
            split_col=args.split_col,
            label_pos=args.label_pos,
            label_neg=args.label_neg,
            image_ext=args.image_ext,
        )
        x_test, y_test = load_data(
            args.test_dir,
            img_size=(224, 224),
            metadata_path=args.metadata,
            image_root=args.image_root,
            split_value=args.test_split,
            image_col=args.image_col,
            label_col=args.label_col,
            split_col=args.split_col,
            label_pos=args.label_pos,
            label_neg=args.label_neg,
            image_ext=args.image_ext,
        )

    model = build_mobilenet(weights=None if args.no_pretrained else "imagenet")
    if args.weights:
        model.load_weights(args.weights)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    if args.eval_only:
        if not args.weights:
            raise ValueError('--weights must be set in eval_only mode')
    else:
        if args.no_aug:
            history = model.fit(
                x_train, y_train,
                batch_size=args.batch_size,
                epochs=args.epochs,
                validation_data=(x_test, y_test),
                shuffle=True,
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
            )

        train_acc = np.array(history.history.get('accuracy', []))
        test_acc = np.array(history.history.get('val_accuracy', []))
        train_loss = np.array(history.history.get('loss', []))
        test_loss = np.array(history.history.get('val_loss', []))

        np.savetxt(f'{args.prefix}-whole-train_acc.txt', train_acc)
        np.savetxt(f'{args.prefix}-whole-test_acc.txt', test_acc)
        np.savetxt(f'{args.prefix}-whole-train_loss.txt', train_loss)
        np.savetxt(f'{args.prefix}-whole-test_loss.txt', test_loss)

        model.save_weights(args.weights_out)

    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(model.predict(x_test, batch_size=args.batch_size), axis=1)
    metrics = compute_metrics(y_true, y_pred)
    append_metrics_csv(args.metrics_csv, args.variant, args.seed, metrics)
    print('Eval metrics:')
    for k, v in metrics.items():
        print(f'{k}: {v}')


if __name__ == '__main__':
    main()
