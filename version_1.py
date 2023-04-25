from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

# Cargar el dataset iris
iris = load_iris()

# Crear un DataFrame a partir de los datos del dataset
df = pd.DataFrame(data=iris['data'], columns=iris['feature_names'])
df['target'] = iris['target']


# Definir la función para predecir con K-Nearest Neighbors (KNN)
def knn_predict():
    # Dividir los datos en conjunto de entrenamiento y conjunto de prueba
    x_train, x_test, y_train, y_test = train_test_split(df[iris['feature_names']], df['target'], test_size=0.3,
                                                        random_state=42)
    # Crear un modelo de KNN con k=3 vecinos
    knn = KNeighborsClassifier(n_neighbors=3)
    # Entrenar el modelo con los datos de entrenamiento
    knn.fit(x_train, y_train)

    # Realizar las predicciones con el conjunto de prueba
    y_pred = knn.predict(x_test)

    # Calcular la precisión del modelo
    accuracy = accuracy_score(y_test, y_pred)
    print("Precisión para el modelo K-Nearest Neighbors (KNN): {:.2f}".format(accuracy))
    # Imprimir la matriz de confusión y el reporte de clasificación
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(" -- -------------------- -- ")


# Definir la función para predecir con Regresión Logística
def log_reg_predict():
    # Dividir los datos en conjunto de entrenamiento y conjunto de prueba
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)
    # Crear un modelo de Regresión Logística
    logreg = LogisticRegression()
    # Entrenar el modelo con los datos de entrenamiento
    logreg.fit(x_train, y_train)

    # Realizar las predicciones con el conjunto de prueba
    y_pred = logreg.predict(x_test)

    # Calcular la precisión del modelo
    accuracy = accuracy_score(y_test, y_pred)
    print("Precisión del modelo de regresión logística: {:.2f}".format(accuracy))


# Definir la función para predecir con Árbol de Decisión
def decision_tree():
    # Dividir los datos en conjunto de entrenamiento y conjunto de prueba
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)
    # Crear un modelo de Árbol de Decisión con una profundidad máxima de 3 niveles
    dt = DecisionTreeClassifier(max_depth=3)
    # Entrenar el modelo con los datos de entrenamiento
    dt.fit(x_train, y_train)

    # Realizar las predicciones con el conjunto de prueba
    y_pred = dt.predict(x_test)

    # Calcular la precisión del modelo
    accuracy = accuracy_score(y_test, y_pred)
    print("Precisión del modelo de árbol de decisión: {:.2f}".format(accuracy))


# Definir la función para predecir con Random Forest
def rand_forrest():
    # Dividir los datos en conjunto de entrenamiento y conjunto de prueba
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)
    # Crear un modelo de Random Forest con 10 árboles
    rf = RandomForestClassifier(n_estimators=10)
    # Entrenar el modelo con los datos de entrenamiento
    rf.fit(x_train, y_train)

    # Realizar las predicciones con el conjunto de prueba
    y_pred = rf.predict(x_test)

    print(" -------------------- ")
    # Calcular la precisión del modelo
    accuracy = accuracy_score(y_test, y_pred)
    print("Precisión del modelo de RandomForrest: {:.2f}".format(accuracy))

    # Realizar una validación cruzada con k=5
    scores = cross_val_score(rf, iris.data, iris.target, cv=5)

    # Imprimir los resultados de la validación
    print("Precisión: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    # Realizar una validación cruzada estratificada con k=5
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(rf, iris.data, iris.target, cv=cv)

    # Imprimir los resultados de la validación
    print("Precisión: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# Ejecutar las funciones para predecir con cada modelo
if __name__ == "__main__":
    knn_predict()
    log_reg_predict()
    decision_tree()
    rand_forrest()
