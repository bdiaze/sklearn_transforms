from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler

# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')

class NormalizarColumnas(BaseEstimator, TransformerMixin):
    def __init__(self, columnas):
        self.columnas = columnas
        
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        data = X.copy()
        scaler = MinMaxScaler()
        data[self.columnas] = scaler.fit_transform(data[self.columnas])
        return data
