import pytest
from rbg.Preprocessing import Standard
import numpy as np
import pandas as pd

#trocar as entradas Data por valores que faça sentido

#escolher o epsylon de maquina
MEpsylon = 10e-9

class TestDataEntry():
    @pytest.mark.parametrize("func", [Standard])
    def test_DataFrame(self,func):
        Data = pd.DataFrame({"coluna1": [1.0, 2.0, 3.0]})
        
        DtFrameResult = func().fit(Data).transform(Data)
        
        assert isinstance(DtFrameResult, np.ndarray)

    @pytest.mark.parametrize("func", [Standard])
    def test_Numpy(self,func):
        Data = np.array([1.0, 2.0, 3.0])

        NpResult = func().fit(Data).transform(Data)

        assert isinstance(NpResult, np.ndarray)

class TestFunctionality():

    def test_Standard(self):
        
        Data = np.random.uniform(0, 100, size=50)
    
        ExpectedResult = (Data - np.mean(Data)) / np.std(Data)
        #testa elemento por elemento da array
        np.testing.assert_allclose(
            Standard().fit(Data).transform(Data), 
            ExpectedResult, 
            atol=MEpsylon,
            err_msg="O calculo divergiu da matriz esperada acima do limite de MEpsylon."
        )
