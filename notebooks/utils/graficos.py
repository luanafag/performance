"""Funções de apoio para os gráficos em Altair.

A parte de PCA e K-means usa as implementações da rbg; aqui só ficam as
tabelas e os gráficos montados a partir delas.
"""
import altair as alt
import numpy as np
import pandas as pd

from rbg.Preprocessing import Standard
from rbg.Unsupervised import KMeans, PCA

from .dados import SEED

CLASS_COLORS = {
    "DERMASON": "#0173B2",
    "SIRA":     "#DE8F05",
    "SEKER":    "#029E73",
    "HOROZ":    "#D55E00",
    "CALI":     "#CC78BC",
    "BARBUNYA": "#CA9161",
    "BOMBAY":   "#FBAFE4",
}

CLUSTER_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
                  "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]


def stylize(chart):
    return (
        chart
        .configure_axis(labelFontSize=10, titleFontSize=11, gridColor="#E5E5E5")
        .configure_legend(symbolOpacity=1, labelFontSize=10, titleFontSize=11)
        .configure_title(fontSize=14)
    )


# ------------------------------------------------------------------- PCA

def normalize(X):
    """Padroniza cada coluna para média 0 e desvio 1."""
    padronizador = Standard()
    padronizador.fit(X)
    return padronizador.transform(X)


def fit_pca(Z, features):
    """Ajusta o PCA com todas as componentes.

    Retorna o modelo, os scores, a tabela de cargas (com a variância explicada
    nas duas últimas linhas) e os vetores de variância explicada e acumulada.
    """
    pca = PCA()
    pca.fit(Z)
    scores = pca.transform(Z)

    var_exp = np.asarray(pca.explained_variance_ratio)
    var_acum = np.cumsum(var_exp)
    nomes = [f"PC{i + 1}" for i in range(len(var_exp))]

    tabela = pd.DataFrame(pca.components.T, index=features, columns=nomes)
    tabela.loc["var. explicada"] = var_exp
    tabela.loc["var. acumulada"] = var_acum
    return pca, scores, tabela.round(3), var_exp, var_acum


def pca_elbow_chart(var_exp, var_acum):
    """Barras com a variância de cada componente e linha com a acumulada."""
    dados = pd.DataFrame({
        "componente": [f"PC{i + 1}" for i in range(len(var_exp))],
        "explicada": var_exp,
        "acumulada": var_acum,
    })
    ordem = dados["componente"].tolist()
    eixo_x = alt.X("componente:N", sort=ordem, title="Componente principal",
                   axis=alt.Axis(labelAngle=0))

    barras = alt.Chart(dados).mark_bar(color="#4C72B0").encode(
        x=eixo_x,
        y=alt.Y("explicada:Q", title="Proporção da variância", axis=alt.Axis(format="%")),
        tooltip=["componente", alt.Tooltip("explicada:Q", format=".1%")],
    )
    linha = alt.Chart(dados).mark_line(color="#C44E52", point=True).encode(
        x=eixo_x,
        y=alt.Y("acumulada:Q"),
        tooltip=["componente", alt.Tooltip("acumulada:Q", format=".1%")],
    )
    corte = alt.Chart(pd.DataFrame({"y": [0.9]})).mark_rule(strokeDash=[4, 4], color="gray").encode(y="y:Q")

    return (barras + linha + corte).properties(
        width=560, height=300,
        title="Variância explicada por componente (linha: acumulada; tracejado: 90%)",
    )


# --------------------------------------------------------------- K-means

def sweep_k(S, ks=range(2, 11)):
    """Roda o K-means para vários valores de K e guarda a WSS de cada um."""
    modelos, wss = {}, {}
    for k in ks:
        modelo = KMeans(k=k, seed=SEED).fit(S)
        modelos[k] = modelo
        wss[k] = modelo.wss
    tabela = pd.DataFrame({"K": list(wss), "WSS": list(wss.values())})
    tabela["queda (%)"] = (-tabela["WSS"].pct_change() * 100).round(1)
    return modelos, wss, tabela


def wss_chart(wss):
    dados = pd.DataFrame({"K": list(wss), "WSS": list(wss.values())})
    return alt.Chart(dados).mark_line(point=True).encode(
        x=alt.X("K:O", title="Número de clusters (K)", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("WSS:Q", title="Soma de quadrados intra-cluster (WSS)", scale=alt.Scale(zero=False)),
        tooltip=["K", alt.Tooltip("WSS:Q", format=",.0f")],
    ).properties(width=560, height=300, title="Gráfico do cotovelo do K-means")


def projection_chart(S, km, cor):
    """Dispersão nas duas primeiras componentes, com os centroides do K-means.

    `cor` é uma tupla (nome do campo, valores por ponto, alt.Color).
    """
    campo, valores, escala_cor = cor
    pontos = pd.DataFrame({"PC1": S[:, 0], "PC2": S[:, 1], campo: valores})
    centroides = pd.DataFrame({
        "PC1": km.centroids[:, 0],
        "PC2": km.centroids[:, 1],
        "cluster": [f"C{i}" for i in range(len(km.centroids))],
    })

    dispersao = alt.Chart(pontos).mark_circle(size=12, opacity=0.4).encode(
        x=alt.X("PC1:Q", title="PC1"),
        y=alt.Y("PC2:Q", title="PC2"),
        color=escala_cor,
    )
    marcas = alt.Chart(centroides).mark_point(
        shape="cross", size=250, color="black", filled=True, opacity=1
    ).encode(x="PC1:Q", y="PC2:Q", tooltip=["cluster"])
    rotulos = alt.Chart(centroides).mark_text(dy=-14, fontWeight="bold").encode(
        x="PC1:Q", y="PC2:Q", text="cluster"
    )

    return (dispersao + marcas + rotulos).properties(
        width=560, height=400,
        title="Projeção nas duas primeiras componentes (× = centroides)",
    )


# --------------------------------------------------------------- árvores

def plot_arvore(tree, feature_names=None):
    """Desenha a estrutura de uma árvore da rbg."""
    info = tree.get_tree_structure_info()

    left = info['children_left']
    right = info['children_right']
    feature = info['feature']
    threshold = info['threshold']
    value = info['value']

    pos = {}
    proxima_folha = [0]

    def posicao(node, depth=0):
        filhos = [c for c in (left[node], right[node]) if c != -1]
        if not filhos:
            x = proxima_folha[0]
            proxima_folha[0] += 1
        else:
            xs = [posicao(c, depth + 1) for c in filhos]
            x = sum(xs) / len(xs)
        pos[node] = (x, -depth)
        return x

    posicao(0)

    nos = []
    for node, (x, y) in pos.items():
        is_leaf = (left[node] == -1 and right[node] == -1)

        if is_leaf:
            val = value[node]
            if isinstance(val, (np.ndarray, list)):
                pred = f"{np.round(val, 2)}"
            else:
                pred = f"{val:.2f}" if isinstance(val, (int, float, np.number)) else f"{val}"
            nos.append({"x": x, "y": y, "texto": f"Pred: {pred}", "pred": pred, "folha": True})
        else:
            feat_idx = feature[node]
            if feature_names is not None and feat_idx < len(feature_names):
                feat_name = feature_names[feat_idx]
            else:
                feat_name = f"X[{feat_idx}]"
            texto = f"{feat_name} ≤ {threshold[node]:.2f}"
            nos.append({"x": x, "y": y, "texto": texto, "pred": "", "folha": False})
    nos = pd.DataFrame(nos)

    arestas = []
    for node, (x, y) in pos.items():
        for child in (left[node], right[node]):
            if child != -1:
                x2, y2 = pos[child]
                arestas.append({"x": x, "y": y, "x2": x2, "y2": y2})
    arestas = pd.DataFrame(arestas)

    eixo_x = alt.X("x:Q", axis=None, scale=alt.Scale(padding=60))
    eixo_y = alt.Y("y:Q", axis=None, scale=alt.Scale(padding=30))

    linhas = alt.Chart(arestas).mark_rule(color="gray").encode(
        x=eixo_x, y=eixo_y, x2="x2:Q", y2="y2:Q"
    )
    decisoes = alt.Chart(nos).transform_filter(
        alt.datum.folha == False
    ).mark_square(size=300, stroke="black", color="lightgray", opacity=1).encode(
        x=eixo_x, y=eixo_y, tooltip=["texto"]
    )
    folhas = alt.Chart(nos).transform_filter(
        alt.datum.folha == True
    ).mark_square(size=300, stroke="black", color="#55A868", opacity=1).encode(
        x=eixo_x, y=eixo_y, tooltip=["texto"],
    )
    rotulos = alt.Chart(nos).mark_text(dy=-18, fontSize=11).encode(
        x=eixo_x, y=eixo_y, text="texto"
    )

    n_folhas = proxima_folha[0]
    return (linhas + decisoes + folhas + rotulos).properties(
        width=min(1600, max(400, n_folhas * 110)),
        height=max(250, (-nos["y"].min() + 1) * 90),
        title="Estrutura da árvore de decisão",
    ).configure_view(stroke=None)


def configurar_altair():
    """Libera bases com mais de 5000 linhas e salva os gráficos como imagem.

    Com o renderizador PNG os notebooks ficam leves e os gráficos aparecem no GitHub.
    """
    alt.data_transformers.disable_max_rows()
    alt.renderers.enable("png", scale_factor=1.5)
