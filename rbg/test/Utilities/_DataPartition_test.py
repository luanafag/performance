'''
#ideai do KFold
# ideias do teste de kfold
# testa se retorna os tamanhos que deveriam retornar
#[(test_indx,train_indx) * k]  = KFold(data_size,k = 5)

for test_indx, train_indx in KFold(data_size,k= 5):
    metodo_generico.fit(X[train_indx],Y[train_indx])

    y_tested = metodo_generico.predict(X[test_indx])


#ideia do LOOCV
#testa se retorna os tamanhos que deveriam retornar
#[(test_indx,train_indx) * data_size] = LOOCV(data_size)
'''

import pytest
import numpy as np

from rbg.ModelSelection import KFold,LOOCV

class TestError:
    
    @pytest.mark.parametrize("k_invalido", [0, 1, 15])
    def test_kfold_k_invalido(self, k_invalido):
        data_size = 10
        #verifica o tamanho correto
        with pytest.raises(ValueError):
            kf = KFold(k = k_invalido)
            list(kf.split(data_size))


class TestRightResults:

    def test_kfold_intersecao_nula_e_uniao_completa(self):
        data_size = 100
        k_esperado = 5
        indice_total_esperado = set(range(data_size))

        kf = KFold(k = k_esperado)
        
        dobras = list(kf.split(data_size))
        
        #se o KFold gerou a quantidade correta de iterações
        assert len(dobras) == k_esperado
        
        for train_indx,test_indx in dobras:
            set_test = set(test_indx)
            set_train = set(train_indx)

            #teste de vazamento de dados
            assert set_train.intersection(set_test) == set()

            #testa se n tem linha de fora
            assert set_train.union(set_test) == indice_total_esperado

    def test_loocv_intersecao_uniao_e_tamanho(self):
        data_size = 50
        indice_total_esperado = set(range(data_size))
        
        loocv = LOOCV()
        
        dobras = list(loocv.split(data_size))
        
        assert len(dobras) == data_size
        
        for train_indx,test_indx in dobras:
            set_test = set(test_indx)
            set_train = set(train_indx)

            #testa o tamanho    
            assert len(test_indx) == 1
            
            #testa vazamento de dados
            assert set_train.intersection(set_test) == set()
            
            #testa se usa todos os dados
            assert set_train.union(set_test) == indice_total_esperado