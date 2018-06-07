# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 1: Regresja liniowa
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# ----------------- TEN PLIK MA POZOSTAC NIEZMODYFIKOWANY ------------------
# --------------------------------------------------------------------------

import numpy as np


def polynomial(x, w):#  funkcja zwracajaca wartosci predykcji y dla zadanego x oraz wektora
#  wartosci parametrów w dla domyslnego modelu wielomianu
    '''
    :param x: wektor argumentow Nx1
    :param w: wektor parametrow (M+1)x1
    :return: wektor wartosci wielomianu w punktach x, Nx1
    '''
    dm = [w[i] * x ** i for i in range(np.shape(w)[0])] #shape zwraca wymiary
    return np.sum(dm, axis=0) #oś - axis
