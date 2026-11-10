"""Carregamento e divisão dos dados, iguais para todos os notebooks.

Todos os modelos de um mesmo problema usam exatamente o mesmo treino e o
mesmo teste, para que as métricas possam ser comparadas.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from rbg.Utilities import train_test_split

SEED = 67

RAIZ = Path(__file__).resolve().parents[2]
PASTA_DADOS = RAIZ / "data"
PASTA_RESULTADOS = RAIZ / "resultados"

# ---------------------------------------------------------------- Rent99

ALVO_RENT = "rent"


def carregar_rent99(remover_rentsqm=True):
    """Lê o rent99. `rentsqm` é rent / area, então vaza o alvo e sai por padrão."""
    rent99 = pd.read_csv(PASTA_DADOS / "rent99.csv")
    if remover_rentsqm:
        rent99 = rent99.drop(columns="rentsqm")
    return rent99


def split_rent99(X, y):
    """80% para treino/validação e 20% para o teste final."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    return train_test_split(X, y, test_size=0.2, random_state=SEED)


# -------------------------------------------------------------- Dry Bean

ALVO_BEAN = "Class"
FEATURES_BEAN = [
    "Area", "Perimeter", "MajorAxisLength", "MinorAxisLength",
    "AspectRatio", "Eccentricity", "ConvexArea", "EquivDiameter",
    "Extent", "Solidity", "Roundness", "Compactness",
    "ShapeFactor1", "ShapeFactor2", "ShapeFactor3", "ShapeFactor4",
]


def carregar_dry_bean(remover_duplicatas=True):
    """Lê o Dry Bean, corrige os nomes das colunas e tira as 68 linhas repetidas."""
    dry_bean = pd.read_csv(PASTA_DADOS / "dry_bean.csv")
    dry_bean = dry_bean.rename(columns={"AspectRation": "AspectRatio", "roundness": "Roundness"})
    if remover_duplicatas:
        dry_bean = dry_bean.drop_duplicates().reset_index(drop=True)
    return dry_bean


def split_dry_bean(dry_bean=None):
    """Divisão estratificada: 20% de cada classe vai para o teste.

    Retorna X_treino, X_teste, y_treino, y_teste (10831 e 2712 linhas).
    """
    if dry_bean is None:
        dry_bean = carregar_dry_bean()

    X = dry_bean[FEATURES_BEAN].to_numpy(dtype=float)
    y = dry_bean[ALVO_BEAN].to_numpy()

    indices_treino, indices_teste = [], []
    for classe in np.unique(y):
        indices = np.flatnonzero(y == classe)
        treino, teste, _, _ = train_test_split(
            indices, indices, test_size=0.2, random_state=SEED
        )
        indices_treino.extend(treino)
        indices_teste.extend(teste)

    return X[indices_treino], X[indices_teste], y[indices_treino], y[indices_teste]


# ------------------------------------------------------------ resultados

def salvar_resultados(nome, tabela):
    """Grava uma tabela de métricas em resultados/<nome>.csv para o relatório final."""
    PASTA_RESULTADOS.mkdir(exist_ok=True)
    caminho = PASTA_RESULTADOS / f"{nome}.csv"
    pd.DataFrame(tabela).to_csv(caminho, index=False)
    return caminho
