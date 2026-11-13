# Estimación de rango etario con UTKFace

Laboratorio en Python 3.10+ con TensorFlow/Keras y transfer learning. Clasifica rostros en **niño (0–12), joven (13–25), adulto (26–59), mayor (60–116)**. Se respeta la descripción de la consigna: el corte en 50 del ejemplo era inconsistente.

## Instalación

Desde la raíz del repositorio:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
```

## Dataset

Descargar y extraer [UTKFace cropped en Kaggle](https://www.kaggle.com/datasets/moritzm00/utkface-cropped) dentro de `data/`. Consultar las condiciones de uso del dataset (investigación académica). No subir las imágenes a Git. No se requieren credenciales para ejecutar el entrenamiento con archivos locales.

Elegir **una sola copia** de UTKFace: si el archivo incluye carpetas `UTKFace` y `utkface_aligned_cropped`, no pasar el directorio padre que contiene ambas. El cargador busca recursivamente y no identifica personas repetidas. El formato esperado es `edad_genero_raza_fecha.jpg` (también admite `.jpg.chip.jpg`). Solo se utiliza la edad.

## Entrenar y evaluar

```bash
.venv/bin/python age_guesser.py train --data data/UTKFace --epochs 5 --output artifacts
```

La primera ejecución descarga los pesos ImageNet desde `storage.googleapis.com`. El entrenamiento puede tardar en CPU. Las imágenes se leen por lotes, se convierten a RGB y se redimensionan a 128×128; el modelo incorpora la transformación a [-1, 1] requerida por MobileNetV2. Se congela la base y se entrena la cabeza de cuatro clases con aumento horizontal y pesos inversos a la frecuencia de clase.

División estratificada: 60% entrenamiento, 20% validación y 20% prueba, semilla 42. Se exigen al menos diez imágenes por clase para poder dividir los datos; ese mínimo permite probar el programa, no garantiza calidad estadística. Early stopping usa solo validación. La prueba se evalúa al terminar. Cada ejecución en una misma carpeta reemplaza sus resultados: elegir otra ruta `--output` para conservar experimentos.

Resultados en `artifacts/`:

- `model.keras`: modelo completo, incluido el preprocesamiento.
- `train.csv`, `val.csv`, `test.csv`: particiones y etiquetas (0 child, 1 young, 2 adult, 3 senior).
- `metrics.json`: accuracy global, precision, recall, F1 y soporte por clase; promedios macro y ponderado.
- `predictions.csv`: predicción y probabilidades por imagen.
- `confusion_matrix.png`, `learning_curve.png`, `history.csv`: visualizaciones y evolución.
- `rejected.json`: archivos inválidos; `run.json`: semilla, versión y tamaños.

Para repetir la evaluación sin entrenar:

```bash
.venv/bin/python age_guesser.py evaluate --model artifacts/model.keras --csv artifacts/test.csv --output artifacts/reevaluation
```

Para evaluar diez rostros conocidos con un modelo ya entrenado, crear un CSV con `path,age,label` y usar el mismo comando. Registrar la edad **al tomar la foto**, no la edad actual, y recortar el rostro. Esas fotos no deben formar parte de entrenamiento o validación. Las probabilidades del modelo no son confianza calibrada.

## Interpretación y entrega

Completar [INFORME.md](INFORME.md) con los resultados reales. Accuracy es la fracción total correcta; precision por clase mide cuántas predicciones de esa clase fueron correctas; recall mide cuántos ejemplos reales de esa clase se recuperaron. Revisar macro-F1 y la matriz por el desbalance de UTKFace. Un soporte cero no permite evaluar esa clase.

Una división aleatoria por imagen puede compartir identidades entre particiones; UTKFace no ofrece aquí una etiqueta fiable de identidad. Esto puede sobreestimar la generalización. Documentar esa limitación, sesgos de representación y errores cerca de los límites de edad. El sistema es un experimento educativo y no determina la edad real ni acredita mayoría de edad.

## Prueba técnica sin dataset

```bash
.venv/bin/python tests/smoke_model.py
```

Ejercita carga, un paso de entrenamiento, guardado/carga del modelo y generación de métricas y gráficos con cuatro imágenes sintéticas y pesos aleatorios. No descarga ImageNet y sus métricas no evalúan estimación de edad.
