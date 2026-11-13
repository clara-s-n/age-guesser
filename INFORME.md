# Informe: estimación de rango etario

## Objetivo y método

Clasificar imágenes faciales en niño (0–12), joven (13–25), adulto (26–59) y mayor (60+). Se utiliza UTKFace cropped, imágenes RGB de 128×128 y MobileNetV2 con pesos ImageNet congelados, pooling global, capa de 128 unidades, dropout 0,3 y salida softmax. Se entrena con Adam y entropía cruzada, aumento horizontal y balanceo por clase.

La partición estratificada usa 60% para entrenamiento, 20% para validación y 20% para prueba. Solo validación participa en early stopping. Registrar semilla, épocas y tamaños desde `run.json`.

## Resultados pendientes de ejecutar con UTKFace

No se han obtenido métricas sobre el dataset real. No se deben sustituir por valores inventados ni por pruebas con imágenes sintéticas.

- Imágenes válidas / rechazadas: pendiente.
- Tamaños train / val / test: pendiente.
- Épocas ejecutadas: pendiente.
- Accuracy global / macro-F1: pendiente (`metrics.json`).

| Clase | Precision | Recall | F1 | Soporte |
|---|---|---|---|---|
| Niño | pendiente | pendiente | pendiente | pendiente |
| Joven | pendiente | pendiente | pendiente | pendiente |
| Adulto | pendiente | pendiente | pendiente | pendiente |
| Mayor | pendiente | pendiente | pendiente | pendiente |

Insertar `learning_curve.png` y `confusion_matrix.png` del experimento. Describir las confusiones más frecuentes y comparar train/val para detectar sobreajuste.

## Conclusiones

Pendientes de las métricas reales. Analizar el desbalance, errores cerca de los cortes de edad y representación del dataset. La división por imagen no garantiza independencia de identidad. Si se añaden diez fotos conocidas, presentar sus resultados aparte: una sola foto cambia la accuracy en diez puntos porcentuales y no permite conclusiones generales.
