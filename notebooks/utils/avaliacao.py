"""Métricas reportadas para todos os modelos, sempre no mesmo formato.

As funções da rbg recebem (previsto, real), nessa ordem.
"""
from rbg.Utilities import Accuracy, F1Score, Recall, MAE, RMSE, RSquare


def metricas_classificacao(modelo, previsto, real):
    """Acurácia, acurácia balanceada (média do recall por classe) e F1 macro."""
    return {
        "modelo": modelo,
        "acuracia": float(Accuracy(previsto, real)),
        "acuracia_balanceada": float(Recall(previsto, real).mean()),
        "f1_macro": float(F1Score(previsto, real).mean()),
    }


def metricas_regressao(modelo, previsto, real):
    """RMSE e MAE em euros, e R²."""
    return {
        "modelo": modelo,
        "RMSE": float(RMSE(previsto, real)),
        "MAE": float(MAE(previsto, real)),
        "R2": float(RSquare(previsto, real)),
    }
