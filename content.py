# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 1: Regresja liniowa
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

import numpy as np
from utils import polynomial


# blad srediokwadratowy
def mean_squared_error(x, y, w):
    '''
    :param x: ciag wejsciowy Nx1
    :param y: ciag wyjsciowy Nx1
    :param w: parametry modelu (M+1)x1
    :return: blad sredniokwadratowy pomiedzy wyjsciami y
    oraz wyjsciami uzyskanymi z wielowamiu o parametrach w dla wejsc x
    '''
    N = len(x)

    err = 1/N * np.sum([(y[n] - polynomial(x[n], w))**2 for n in range(N)]) #n - numer przykladu ze zbioru x
    return err
# liczy maczierz sigma w pliku
def design_matrix(x_train, M):
    '''
    :param x_train: ciag treningowy Nx1
    :param M: stopien wielomianu 0,1,2,...
    :return: funkcja wylicza Design Matrix Nx(M+1) dla wielomianu rzedu M
    '''
    N = len(x_train)
    import numpy as np
    design_matrix = np.zeros(shape=(N, M+1)) #zeros wypelnia zerami matryce
    for i in range(N):
        for j in range(M+1):
            design_matrix[i, j] = x_train[i] ** j

    return design_matrix

def least_squares(x_train, y_train, M): #rozwiazanie liniowego zadania najmniejszych kwadratow
    '''
    :param x_train: ciag treningowy wejscia Nx1
    :param y_train: ciag treningowy wyjscia Nx1
    :param M: rzad wielomianu
    :return: funkcja zwraca krotke (w,err), gdzie w sa parametrami dopasowanego wielomianu, a err blad sredniokwadratowy
    dopasowania
    '''
    sigma = design_matrix(x_train, M)

    w = sigma.transpose()#(2) wzor z polecenia
    w = w.dot(sigma)
    from numpy.linalg import inv
    w = inv(w)#do potegi -1

    w = w.dot(sigma.transpose())
    w = w.dot(y_train)
    err = mean_squared_error(x_train, y_train, w) #(2)
    return w, err


def regularized_least_squares(x_train, y_train, M, regularization_lambda):
    '''
    :param x_train: ciag treningowy wejscia Nx1
    :param y_train: ciag treningowy wyjscia Nx1
    :param M: rzad wielomianu
    :param regularization_lambda: parametr regularyzacji
    :return: funkcja zwraca krotke (w,err), gdzie w sa parametrami dopasowanego wielomianu zgodnie z kryterium z regularyzacja l2,
    a err blad sredniokwadratowy dopasowania
    '''
    sigma = design_matrix(x_train, M)

    w = sigma.transpose()
    w = w.dot(sigma)
    w += np.dot(regularization_lambda, np.eye(M+1)) #eye zwraca tablice, na przekatnej sa jedynki
    from numpy.linalg import inv
    w = inv(w)

    w = w.dot(sigma.transpose())
    w = w.dot(y_train)
    err = mean_squared_error(x_train, y_train, w)

    return w, err


def model_selection(x_train, y_train, x_val, y_val, M_values):
    '''
    :param x_train: ciag treningowy wejscia Nx1
    :param y_train: ciag treningowy wyjscia Nx1
    :param x_val: ciag walidacyjny wejscia Nx1
    :param y_val: ciag walidacyjny wyjscia Nx1
    :param w: parametry modelu (M+1)x1
    :param M_values: tablica stopni wielomianu, ktore maja byc sprawdzone
    :return: funkcja zwraca krotke (w,train_err,val_err), gdzie w sa parametrami modelu, ktory najlepiej generalizuje dane,
    tj. daje najmniejszy blad na ciagu walidacyjnym, train_err i val_err to bledy na sredniokwadratowe na ciagach treningowym
    i walidacyjnym
    '''

    models = [] #lista modeli
    for M in M_values:
        w, train_err = least_squares(x_train, y_train, M)
        val_err = mean_squared_error(x_val, y_val, w)
        models.append({"w": w, "train_err": train_err, "val_err": val_err})

    # f = lambda x: x**2
    # f(3) = 9
    min_model = min(models, key=lambda model: model["val_err"])
    return (min_model["w"], min_model["train_err"], min_model["val_err"])

def regularized_model_selection(x_train, y_train, x_val, y_val, M, lambda_values):
    '''
    :param x_train: ciag treningowy wejscia Nx1
    :param y_train: ciag treningowy wyjscia Nx1
    :param x_val: ciag walidacyjny wejscia Nx1
    :param y_val: ciag walidacyjny wyjscia Nx1
    :param M: stopien wielomianu
    :param lambda_values: lista ze wartosciami roznych parametrow regularyzacji
    :return: funkcja zwraca krotke (w,train_err,val_err,regularization_lambda), gdzie w sa parametrami modelu, ktory najlepiej generalizuje dane,
    tj. daje najmniejszy blad na ciagu walidacyjnym. Wielomian dopasowany jest wg kryterium z regularyzacja. train_err i val_err to
    bledy na sredniokwadratowe na ciagach treningowym i walidacyjnym. regularization_lambda to najlepsza wartosc parametru regularyzacji
    '''
    models = []
    for regularization_lambda in lambda_values:
        w, train_err = regularized_least_squares(x_train, y_train, M, regularization_lambda)
        val_err = mean_squared_error(x_val, y_val, w)
        models.append((w, train_err, val_err, regularization_lambda))

    return min(models, key=lambda model: model[2])
