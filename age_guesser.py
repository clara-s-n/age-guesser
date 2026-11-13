"""Entrenamiento y evaluación reproducible de rangos etarios con UTKFace."""
import argparse
import json
from pathlib import Path

CLASSES = ('child', 'young', 'adult', 'senior')


def categorize_age(age):
    if age < 0 or age > 116:
        raise ValueError(f'Edad fuera del rango UTKFace: {age}')
    return 0 if age <= 12 else 1 if age <= 25 else 2 if age <= 59 else 3


def scan_dataset(root):
    from PIL import Image
    rows, rejected = [], []
    for path in sorted(Path(root).rglob('*')):
        if path.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
            continue
        try:
            age = int(path.name.split('_')[0])
            label = categorize_age(age)
            with Image.open(path) as image:
                image.verify()
            rows.append({'path': str(path.resolve()), 'age': age, 'label': label})
        except (ValueError, OSError, SyntaxError) as error:
            rejected.append({'path': str(path), 'error': str(error)})
    if not rows:
        raise ValueError('No se encontraron imágenes válidas con edad en el nombre.')
    return rows, rejected


def dependencies():
    import tensorflow as tf
    import numpy as np
    import pandas as pd
    return tf, np, pd


def dataset(frame, batch, training=False, seed=42):
    tf, _, _ = dependencies()
    ds = tf.data.Dataset.from_tensor_slices((frame.path.to_numpy(), frame.label.to_numpy()))
    if training:
        ds = ds.shuffle(len(frame), seed=seed)
    def decode(path, label):
        image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        # MobileNetV2 espera RGB en [-1, 1], no imágenes divididas por 255.
        return tf.image.resize(tf.cast(image, tf.float32), (128, 128)), label
    return ds.map(decode, num_parallel_calls=2).batch(batch).prefetch(1)


def build_model(weights='imagenet'):
    tf, _, _ = dependencies()
    base = tf.keras.applications.MobileNetV2(weights=weights, include_top=False,
                                           input_shape=(128, 128, 3))
    base.trainable = False
    inputs = tf.keras.Input((128, 128, 3))
    x = tf.keras.layers.RandomFlip('horizontal')(inputs)
    x = tf.keras.layers.Rescaling(1 / 127.5, offset=-1)(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(.3)(x)
    model = tf.keras.Model(inputs, tf.keras.layers.Dense(4, activation='softmax')(x))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def evaluate(model, frame, output, batch):
    _, np, pd = dependencies()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
    probabilities = model.predict(dataset(frame, batch), verbose=0)
    predicted = probabilities.argmax(axis=1)
    report = classification_report(frame.label, predicted, labels=list(range(4)),
                                   target_names=CLASSES, output_dict=True, zero_division=0)
    report['accuracy'] = float(np.mean(frame.label.to_numpy() == predicted))
    (output / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    result = frame.copy()
    result['predicted'] = [CLASSES[i] for i in predicted]
    for index, name in enumerate(CLASSES):
        result[f'probability_{name}'] = probabilities[:, index]
    result.to_csv(output / 'predictions.csv', index=False)
    matrix = confusion_matrix(frame.label, predicted, labels=list(range(4)))
    ConfusionMatrixDisplay(matrix, display_labels=CLASSES).plot()
    plt.tight_layout()
    plt.savefig(output / 'confusion_matrix.png')
    plt.close()
    print(json.dumps(report, indent=2))


def train(args):
    tf, _, pd = dependencies()
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np
    rows, rejected = scan_dataset(args.data)
    frame = pd.DataFrame(rows)
    counts = frame.label.value_counts()
    if any(counts.get(i, 0) < 10 for i in range(4)):
        raise ValueError('Se requieren al menos 10 imágenes de cada rango para dividir train/val/test.')
    # Test queda reservado: no se usa para selección de modelo ni early stopping.
    rest, test = train_test_split(frame, test_size=.2, stratify=frame.label, random_state=args.seed)
    training, validation = train_test_split(rest, test_size=.25, stratify=rest.label, random_state=args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'rejected.json').write_text(json.dumps(rejected, indent=2), encoding='utf-8')
    for name, split in [('train', training), ('val', validation), ('test', test)]:
        split.to_csv(output / f'{name}.csv', index=False)
    tf.keras.utils.set_random_seed(args.seed)
    model = build_model()
    weights = compute_class_weight('balanced', classes=np.arange(4), y=training.label.to_numpy())
    history = model.fit(dataset(training, args.batch_size, True, args.seed),
                        validation_data=dataset(validation, args.batch_size), epochs=args.epochs,
                        class_weight=dict(enumerate(weights)),
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3,
                                                                     restore_best_weights=True)])
    model.save(output / 'model.keras')
    pd.DataFrame(history.history).to_csv(output / 'history.csv', index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='validation')
    plt.xlabel('Época (desde 0)')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(output / 'learning_curve.png')
    plt.close()
    (output / 'run.json').write_text(json.dumps({'seed': args.seed, 'classes': CLASSES,
        'tensorflow': tf.__version__, 'epochs_requested': args.epochs,
        'epochs_run': len(history.history['loss']), 'split_sizes': {
            'train': len(training), 'val': len(validation), 'test': len(test)}}, indent=2))
    evaluate(model, test, output, args.batch_size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    fit = sub.add_parser('train')
    fit.add_argument('--data', required=True)
    fit.add_argument('--output', default='artifacts')
    fit.add_argument('--epochs', type=int, default=5)
    fit.add_argument('--batch-size', type=int, default=32)
    fit.add_argument('--seed', type=int, default=42)
    check = sub.add_parser('evaluate')
    check.add_argument('--model', required=True)
    check.add_argument('--csv', required=True, help='CSV con columnas path,age,label; rutas absolutas')
    check.add_argument('--output', default='artifacts/evaluation')
    check.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    if args.batch_size < 1 or (args.command == 'train' and args.epochs < 1):
        parser.error('batch-size y epochs deben ser positivos')
    if args.command == 'train':
        train(args)
    else:
        tf, _, pd = dependencies()
        frame = pd.read_csv(args.csv)
        if frame.empty or any(categorize_age(a) != label for a, label in zip(frame.age, frame.label)):
            parser.error('CSV vacío o etiquetas incompatibles con las edades')
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        evaluate(tf.keras.models.load_model(args.model), frame, output, args.batch_size)


if __name__ == '__main__':
    main()
