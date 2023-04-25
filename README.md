# iris-classification-exercise

## version_1:

Este código utiliza diferentes modelos de aprendizaje automático (K-Nearest Neighbors, Regresión Logística, Árbol de Decisión y Random Forest) para predecir la especie de una flor iris a partir de sus características (longitud y ancho de pétalos y sépalos). Se utiliza el dataset iris, que es un conjunto de datos muy conocido en el mundo del aprendizaje automático.

Para cada modelo, se divide el conjunto de datos en un conjunto de entrenamiento y un conjunto de prueba, se entrena el modelo con el conjunto de entrenamiento y se hacen predicciones con el conjunto de prueba. Luego se calcula la precisión del modelo utilizando la función de precisión de sklearn. Además, se realizan validaciones cruzadas para evaluar la precisión del modelo de manera más robusta.

El código también imprime la matriz de confusión y el reporte de clasificación para el modelo KNN. Esto proporciona información adicional sobre la calidad de las predicciones del modelo.
