"""Baixa os dois conjuntos de dados e salva os CSVs brutos nesta pasta.

Os CSVs já estão versionados; este script existe para que qualquer pessoa
consiga regenerá-los a partir das fontes originais.

    pip install requests pyreadr ucimlrepo
    python data/baixar_dados.py
"""
import io
import tarfile
import tempfile
from pathlib import Path

import pyreadr
import requests
from ucimlrepo import fetch_ucirepo

PASTA = Path(__file__).resolve().parent
URLS_GAMLSS = [
    "https://cran.r-project.org/src/contrib/gamlss.data_6.0-7.tar.gz",
    "https://cran.r-project.org/src/contrib/Archive/gamlss.data/gamlss.data_6.0-7.tar.gz",
]


def baixar_rent99():
    for url in URLS_GAMLSS:
        resposta = requests.get(url, timeout=60)
        if resposta.ok:
            break
    resposta.raise_for_status()

    pacote = tarfile.open(fileobj=io.BytesIO(resposta.content), mode="r:gz")
    rda = pacote.extractfile("gamlss.data/data/rent99.rda").read()
    with tempfile.NamedTemporaryFile(suffix=".rda") as tmp:
        tmp.write(rda)
        tmp.flush()
        rent99 = list(pyreadr.read_r(tmp.name).values())[0]

    rent99.to_csv(PASTA / "rent99.csv", index=False)
    print("rent99.csv:", rent99.shape)


def baixar_dry_bean():
    base = fetch_ucirepo(id=602)
    dry_bean = base.data.features.join(base.data.targets)
    dry_bean.to_csv(PASTA / "dry_bean.csv", index=False)
    print("dry_bean.csv:", dry_bean.shape)


if __name__ == "__main__":
    baixar_rent99()
    baixar_dry_bean()
