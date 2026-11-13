import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tempfile
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from age_guesser import build_model, dataset, evaluate, scan_dataset

tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    for age in (8, 20, 40, 70):
        Image.fromarray(np.random.default_rng(age).integers(0, 256, (128,128,3), dtype=np.uint8)).save(root / f'{age}_0_0_test.jpg')
    rows, rejected = scan_dataset(root)
    frame = pd.DataFrame(rows)
    model = build_model(weights=None)
    history = model.fit(dataset(frame, 4), epochs=1, verbose=0)
    assert np.isfinite(history.history['loss'][0])
    model.save(root / 'model.keras')
    restored = tf.keras.models.load_model(root / 'model.keras')
    evaluate(restored, frame, root, 4)
    assert (root / 'confusion_matrix.png').is_file()
    assert len(pd.read_csv(root / 'predictions.csv')) == 4
    print('SMOKE OK: decode, train, save/reload, evaluate, reports (synthetic data, random weights)')
