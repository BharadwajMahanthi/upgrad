import os
import json
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import config as tf_config

warnings.filterwarnings("ignore", category=FutureWarning)

# Centralized Config
import config

# ✅ MobileNetV2 preprocessing
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

try:
    from keras_tuner.tuners import RandomSearch
    from keras_tuner.engine.hyperparameters import HyperParameters
except ImportError:
    try:
        from keras_tuner import RandomSearch, HyperParameters
    except ImportError:
        import keras_tuner as kt
        RandomSearch = kt.RandomSearch
        HyperParameters = kt.HyperParameters

from sklearn.metrics import confusion_matrix, classification_report

layers = tf.keras.layers
models = tf.keras.models
optimizers = tf.keras.optimizers
callbacks_lib = tf.keras.callbacks
applications = tf.keras.applications
regularizers = tf.keras.regularizers


# ============================================================
# ENVIRONMENT SETUP
# ============================================================
def setup_environment():
    print("[INFO] Setting up environment...")

    os.environ["PYTHONHASHSEED"] = str(config.SEED)
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    tf.random.set_seed(config.SEED)

    gpus = tf_config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf_config.experimental.set_memory_growth(gpu, True)
            print("[INFO] GPU detected ✅ Memory growth enabled.")
        except RuntimeError as e:
            print(f"[WARNING] Could not enable memory growth: {e}")
    else:
        print("[INFO] No GPU detected → running on CPU.")

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    print(f"[INFO] Output directory ready: {config.OUTPUT_DIR}")


# ============================================================
# DATA ANALYSIS (OPTIONAL)
# ============================================================
def analyze_data(dataset_dir: str):
    print(f"[INFO] Analyzing dataset directory: {dataset_dir}")

    if not os.path.exists(dataset_dir):
        print(f"[WARNING] Dataset directory not found: {dataset_dir}")
        return

    file_types = {}
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            file_types[ext] = file_types.get(ext, 0) + 1

    df = pd.DataFrame(list(file_types.items()), columns=["Extension", "Count"]).sort_values(
        by="Count", ascending=False
    )

    print("\n[INFO] File Type Counts:")
    print(df.to_string(index=False))


# ============================================================
# STRONG AUGMENTATION FOR REAL WORLD (GLASS / REFLECTION / WEBCAM)
# ============================================================
def _random_glare_mask(h, w):
    """
    Creates a random elliptical glare-like mask in [0,1]
    """
    # random center
    cx = tf.random.uniform([], 0.2, 0.8) * tf.cast(w, tf.float32)
    cy = tf.random.uniform([], 0.2, 0.8) * tf.cast(h, tf.float32)

    # random ellipse radii
    rx = tf.random.uniform([], 0.08, 0.25) * tf.cast(w, tf.float32)
    ry = tf.random.uniform([], 0.08, 0.25) * tf.cast(h, tf.float32)

    # build coordinate grid
    x = tf.linspace(0.0, tf.cast(w - 1, tf.float32), w)
    y = tf.linspace(0.0, tf.cast(h - 1, tf.float32), h)
    xx, yy = tf.meshgrid(x, y)

    # ellipse equation
    ellipse = ((xx - cx) ** 2) / (rx ** 2 + 1e-6) + ((yy - cy) ** 2) / (ry ** 2 + 1e-6)
    mask = tf.exp(-ellipse * 2.5)  # smooth falloff

    # normalize 0..1
    mask = tf.clip_by_value(mask, 0.0, 1.0)
    mask = tf.expand_dims(mask, axis=-1)
    return mask


def add_glass_reflection(img):
    """
    Simulate glass reflection by adding a bright glare patch
    and slightly washing out colors.
    """
    h = tf.shape(img)[0]
    w = tf.shape(img)[1]

    glare = _random_glare_mask(h, w)

    # strength + tint
    strength = tf.random.uniform([], 0.10, 0.35)
    glare_color = tf.random.uniform([1, 1, 3], 0.85, 1.0)

    # add glare
    img = img + glare * strength * glare_color

    # slight washout (like glass)
    wash = tf.random.uniform([], 0.00, 0.08)
    img = img * (1.0 - wash) + wash

    return tf.clip_by_value(img, 0.0, 1.0)


def add_gaussian_noise(img):
    std = tf.random.uniform([], 0.0, 0.06)
    noise = tf.random.normal(tf.shape(img), mean=0.0, stddev=std)
    img = img + noise
    return tf.clip_by_value(img, 0.0, 1.0)


def random_jpeg_compression(img):
    """
    Simulate webcam/compression artifacts
    """
    quality = tf.random.uniform([], 35, 100, dtype=tf.int32)

    # Convert to uint8
    x = tf.cast(img * 255.0, tf.uint8)

    # encode/decode jpeg
    x = tf.io.encode_jpeg(x, quality=quality)
    x = tf.io.decode_jpeg(x, channels=3)

    # back to float
    x = tf.cast(x, tf.float32) / 255.0
    return tf.clip_by_value(x, 0.0, 1.0)


def random_motion_blur(img):
    """
    Light motion blur using separable 1D blur kernel
    """
    k = tf.random.uniform([], 1, 6, dtype=tf.int32)  # 1..5
    if k <= 1:
        return img

    # Build 1D kernel
    kernel = tf.ones([k], dtype=tf.float32) / tf.cast(k, tf.float32)

    # blur horizontal or vertical randomly
    if tf.random.uniform([]) > 0.5:
        # horizontal
        kernel2d = tf.reshape(kernel, [1, k, 1, 1])
    else:
        # vertical
        kernel2d = tf.reshape(kernel, [k, 1, 1, 1])

    # depthwise conv: apply same blur per channel
    kernel2d = tf.tile(kernel2d, [1, 1, 3, 1])

    img4 = tf.expand_dims(img, axis=0)  # (1,H,W,3)
    img_blur = tf.nn.depthwise_conv2d(
        img4, kernel2d, strides=[1, 1, 1, 1], padding="SAME"
    )
    img_blur = tf.squeeze(img_blur, axis=0)
    return tf.clip_by_value(img_blur, 0.0, 1.0)


def color_jitter(img):
    """
    Robust lighting / tint jitter (important for fire/smoke behind glass)
    """
    img = tf.image.random_brightness(img, max_delta=0.20)
    img = tf.image.random_contrast(img, lower=0.70, upper=1.35)
    img = tf.image.random_saturation(img, lower=0.75, upper=1.35)
    img = tf.image.random_hue(img, max_delta=0.05)
    return tf.clip_by_value(img, 0.0, 1.0)


def strong_augment(img, label):
    """
    Takes float image in [0,1] and applies realistic disturbances.
    """
    # Randomly apply each augmentation
    if tf.random.uniform([]) < 0.70:
        img = color_jitter(img)

    if tf.random.uniform([]) < 0.55:
        img = add_gaussian_noise(img)

    if tf.random.uniform([]) < 0.35:
        img = random_motion_blur(img)

    if tf.random.uniform([]) < 0.40:
        img = random_jpeg_compression(img)

    # ✅ Most important: reflections/glare simulation
    if tf.random.uniform([]) < 0.55:
        img = add_glass_reflection(img)

    return img, label


# ============================================================
# DATA PIPELINE (tf.data) - STRONGER THAN ImageDataGenerator
# ============================================================
def create_datasets():
    if not os.path.exists(config.TRAIN_DIR) or not os.path.exists(config.TEST_DIR):
        raise FileNotFoundError(
            f"TRAIN_DIR or TEST_DIR missing.\nTRAIN_DIR={config.TRAIN_DIR}\nTEST_DIR={config.TEST_DIR}"
        )

    print("[INFO] Creating tf.data datasets...")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        config.TRAIN_DIR,
        labels="inferred",
        label_mode="categorical",
        class_names=config.CLASSES,  # ✅ ensures stable class order
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        seed=config.SEED,
        validation_split=config.VAL_SPLIT,
        subset="training",
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        config.TRAIN_DIR,
        labels="inferred",
        label_mode="categorical",
        class_names=config.CLASSES,
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        seed=config.SEED,
        validation_split=config.VAL_SPLIT,
        subset="validation",
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        config.TEST_DIR,
        labels="inferred",
        label_mode="categorical",
        class_names=config.CLASSES,
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        seed=config.SEED,
    )

    AUTOTUNE = tf.data.AUTOTUNE

    # Convert to float and normalize to [0,1] before augmentations
    def to_float(img, label):
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    train_ds = train_ds.map(to_float, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(to_float, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.map(to_float, num_parallel_calls=AUTOTUNE)

    # ✅ Apply strong augmentation ONLY to training
    train_ds = train_ds.map(strong_augment, num_parallel_calls=AUTOTUNE)

    # ✅ Apply MobileNetV2 preprocess_input at the end (expects 0..255 but works fine with float too)
    def mobilenet_preprocess(img, label):
        # preprocess_input expects float32, works with 0..255 or 0..1 but best is 0..255
        img = img * 255.0
        img = preprocess_input(img)
        return img, label

    train_ds = train_ds.map(mobilenet_preprocess, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(mobilenet_preprocess, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.map(mobilenet_preprocess, num_parallel_calls=AUTOTUNE)

    # Prefetch for speed
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)

    # Class indices mapping (consistent)
    class_indices = {name: i for i, name in enumerate(config.CLASSES)}
    return train_ds, val_ds, test_ds, class_indices


def compute_class_weights_from_ds(train_ds):
    """
    Computes class weights from labels in the dataset.
    Helps when class imbalance exists (usually true in fire/smoke).
    """
    counts = np.zeros(len(config.CLASSES), dtype=np.int64)

    for _, y in train_ds:
        y_np = y.numpy()
        labels = np.argmax(y_np, axis=1)
        for lab in labels:
            counts[lab] += 1

    total = counts.sum()
    weights = {}

    for i in range(len(counts)):
        # weight inversely proportional to class frequency
        weights[i] = float(total / (len(counts) * (counts[i] + 1e-6)))

    print("[INFO] Class counts:", counts.tolist())
    print("[INFO] Class weights:", weights)
    return weights


# ============================================================
# MODEL BUILDER (KERAS TUNER)
# ============================================================
def build_model(hp: HyperParameters) -> tf.keras.Model:
    base_model = applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(config.IMG_SIZE[0], config.IMG_SIZE[1], 3),
    )

    base_model.trainable = False  # phase-1 frozen

    dense_units = int(hp.Choice("dense_units", values=[64, 128, 256]) or 128)
    dropout_rate = float(hp.Float("dropout", min_value=0.25, max_value=0.65, step=0.10) or 0.4)
    lr = float(hp.Float("learning_rate", min_value=1e-5, max_value=2e-3, sampling="log") or 3e-4)

    model = models.Sequential(
        [
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(dense_units, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            layers.Dense(len(config.CLASSES), activation="softmax", dtype="float32"),
        ]
    )

    # ✅ label smoothing reduces overconfident wrong predictions (safer)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.08)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss=loss_fn,
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def fine_tune(model, fine_tune_layers=40, lr=1e-5):
    """
    Phase-2 fine-tuning: unfreeze top layers of MobileNetV2.
    """
    base_model = model.layers[0]
    base_model.trainable = True

    # freeze early layers, unfreeze last N layers
    for layer in base_model.layers[:-fine_tune_layers]:
        layer.trainable = False

    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss=loss_fn,
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    return model


# ============================================================
# PLOTTING + REPORTS
# ============================================================
def plot_training_curves(history, out_path: str):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history.get("loss", []), label="train_loss")
    plt.plot(history.history.get("val_loss", []), label="val_loss")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history.get("accuracy", []), label="train_acc")
    plt.plot(history.history.get("val_accuracy", []), label="val_acc")
    plt.title("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def evaluate_and_save_reports(model, test_ds):
    print("[INFO] Running evaluation on test set...")

    y_true = []
    y_pred = []

    for x_batch, y_batch in test_ds:
        preds = model.predict(x_batch, verbose=0)
        y_true.extend(np.argmax(y_batch.numpy(), axis=1).tolist())
        y_pred.extend(np.argmax(preds, axis=1).tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=config.CLASSES,
        yticklabels=config.CLASSES,
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    cm_path = os.path.join(config.OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[INFO] Confusion matrix saved: {cm_path}")

    report = classification_report(y_true, y_pred, target_names=config.CLASSES)
    print("\n[INFO] Classification Report:\n")
    print(report)

    report_path = os.path.join(config.OUTPUT_DIR, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(str(report))
    print(f"[INFO] Classification report saved: {report_path}")


# ============================================================
# MAIN
# ============================================================
def main():
    setup_environment()

    analyze_data(config.TRAIN_DIR)
    analyze_data(config.TEST_DIR)

    train_ds, val_ds, test_ds, class_indices = create_datasets()

    # Save mapping for deployment
    class_map_path = os.path.join(config.OUTPUT_DIR, config.CLASS_MAP_FILENAME)
    with open(class_map_path, "w", encoding="utf-8") as f:
        json.dump(class_indices, f, indent=4)

    print(f"[INFO] Saved class mapping to: {class_map_path}")
    print(f"[INFO] class_indices = {class_indices}")

    class_weights = compute_class_weights_from_ds(train_ds)

    tuner = RandomSearch(
        build_model,
        objective="val_accuracy",
        max_trials=config.MAX_TUNER_TRIALS,
        overwrite=True,
        directory=os.path.join(config.OUTPUT_DIR, "kt_search"),
        project_name="fire_tuning",
    )

    callbacks = [
        callbacks_lib.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        callbacks_lib.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
    ]

    print("[INFO] Starting hyperparameter tuning...")
    tuner.search(
        train_ds,
        validation_data=val_ds,
        epochs=config.TUNER_EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    best_model = tuner.get_best_models(num_models=1)[0]
    print("[INFO] Best model selected from tuning ✅")

    # ======================================================
    # PHASE 1: Train head (base frozen)
    # ======================================================
    best_model_path = os.path.join(config.OUTPUT_DIR, config.MODEL_FILENAME)

    checkpoint = callbacks_lib.ModelCheckpoint(
        best_model_path,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )

    print("[INFO] Phase 1: Training classifier head (base frozen)...")
    history1 = best_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.TRAIN_EPOCHS,
        callbacks=[checkpoint] + callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    phase1_plot = os.path.join(config.OUTPUT_DIR, "training_plot_phase1.png")
    plot_training_curves(history1, phase1_plot)
    print(f"[INFO] Phase 1 curves saved: {phase1_plot}")

    # ======================================================
    # PHASE 2: Fine-tune top MobileNet layers
    # ======================================================
    print("[INFO] Phase 2: Fine-tuning top backbone layers...")
    fine_tuned_model = tf.keras.models.load_model(best_model_path)

    fine_tuned_model = fine_tune(
        fine_tuned_model,
        fine_tune_layers=40,
        lr=1e-5
    )

    fine_tune_checkpoint_path = os.path.join(config.OUTPUT_DIR, "fire_detection_finetuned.keras")
    fine_checkpoint = callbacks_lib.ModelCheckpoint(
        fine_tune_checkpoint_path,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )

    history2 = fine_tuned_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=max(5, config.TRAIN_EPOCHS // 2),
        callbacks=[fine_checkpoint] + callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    phase2_plot = os.path.join(config.OUTPUT_DIR, "training_plot_phase2.png")
    plot_training_curves(history2, phase2_plot)
    print(f"[INFO] Phase 2 curves saved: {phase2_plot}")

    print(f"[INFO] Fine-tuned model saved to: {fine_tune_checkpoint_path}")

    # Evaluate final fine-tuned model
    final_model = tf.keras.models.load_model(fine_tune_checkpoint_path)
    evaluate_and_save_reports(final_model, test_ds)

    tf.keras.backend.clear_session()
    print("[INFO] Done ✅")


if __name__ == "__main__":
    main()
