from libsvm.svmutil import *
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from Classifier import *

def test_binary_prediction(predictions, real, threshold):
    """Comprueba la precision del modelo binario comparando las predicciones 
    con los datos reales
    predictions: Arreglo de predicciones
    real: Arreglo de datos reales
    threshold: Umbral classifier partir del que se considera que una predicción binaria
        es o no es la clase.
    """
    pred = list(map(lambda x: x[0], predictions))
    actual = list(map(lambda x: x[0], real))
    pred_and_real = list(zip(pred, actual))
    
    results = list(map(lambda data: (data[0] >= threshold, data[1]), pred_and_real))
    results = pd.DataFrame(results, columns=['prediction', 'real'])
    
    false_positives = len(results[(results['prediction']==True) & (results['real']==False)])
    false_negatives = len(results[(results['prediction']==False) & (results['real']==True)])
    return [false_positives, false_negatives]


iris_test = pd.read_csv("iris data\iris test data.csv")
iris_train = pd.read_csv("iris data\iris train data.csv")


Y1 = np.array(iris_test["species"])
X1 = np.array(iris_test.drop("species",axis=1))
Y2 = np.array(iris_train["species"])
X2 = np.array(iris_train.drop("species",axis=1))


IRIS = pd.read_csv("iris data\only_Iris-versicolor_train.csv")
def accuracy(false_positives,false_negatives,total):
    return (total - (false_positives+false_negatives))/total

def classify(specie,it):
	threshold=0.85
	classifier = Classifier()
	classifier.binary(iris_train, specie, [], iterations=it, epsilon=0.0004, alpha=0.001)	
	predictions = classifier.network.predict(iris_test.drop("species",axis=1)).values.tolist()
	Y = classifier.modify_Y_binary(Y1,specie)
	test = test_binary_prediction(predictions,Y, threshold)
	pred = list(map(lambda x: specie if (x[0] >= threshold) else 0, predictions))
	print(test)
	accurate = accuracy(test[0],test[1],len(iris_test))
	print("La exactitud para ",specie,"es = ",accurate)
	return pred, accurate

setosa, setosa_accuracy = classify("Iris-setosa",100000)
versicolor, versicolor_accuracy = classify("Iris-versicolor",100000)
virginica, virginica_accuracy = classify("Iris-virginica",100000)

#Se considera el kernel lineal debido a que los datos son linealmente separables
model0 = SVC().fit(X2, Y2)
P0 = model0.predict(X1)

model = SVC(kernel="linear").fit(X2, Y2)
P1 = model.predict(X1)
'''
P1_setosa = list(map(lambda x: x if x == "Iris-setosa" else 0, P1))
P1_versicolor= list(map(lambda x: x if x == "Iris-versicolor" else 0, P1))
P1_virginica = list(map(lambda x: x if x == "Iris-virginica" else 0, P1))
print(P1_setosa == setosa)
print(P1_versicolor == versicolor)
print(P1_virginica == virginica)
print(P1 == Y1)
'''
#Se considera el kernel polinomial porque logra alcanzar una accuracy de 100% para el testeo

model2 = SVC(kernel="poly").fit(X2, Y2)
P2 = model2.predict(X1)

SVC_accuracy_no_kernel = accuracy_score(Y1,P0)
SVC_accuracy = accuracy_score(Y1, P1)
SVC_accuracy_kernel = accuracy_score(Y1,P2)
print("Exactitud para SVC ",SVC_accuracy_no_kernel)
print("Exactitud para SVC kernel lineal",SVC_accuracy)
print("Exactitud para SVC kernel polinomial",SVC_accuracy_kernel)

