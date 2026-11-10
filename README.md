# Aplicação da RBG

Parte II do projeto final de Aprendizado de Máquina I (IMPA Tech, 2026.2). Aqui usamos a [rbg](https://github.com/fragadmsc/rbg), a biblioteca que o grupo implementou só com NumPy na Parte I, para resolver dois problemas com dados reais:

- **Regressão**: prever o aluguel mensal de imóveis em Munique (Rent99).
- **Classificação**: identificar o tipo de um grão de feijão a partir de medidas da sua imagem (Dry Bean).

Para cada problema fizemos análise exploratória, pré-processamento, comparação de modelos com validação, avaliação final em um conjunto de teste separado e interpretação dos resultados. Redes neurais entram na comparação dos dois problemas.

## Como rodar

Precisa de Python 3.12 ou mais novo.

```bash
pip install -r requirements.txt
```

```bash
jupyter lab
```

Os notebooks 01, 03 e 05 estão salvos com as saídas; o 02, o 04 e o relatório precisam ser executados. Para reexecutar, a ordem é a da numeração; o `relatorio_final.ipynb` lê as tabelas que os notebooks 02, 04 e 05 gravam em `resultados/`, então ele vem por último. Os notebooks 02 e 04 levam alguns minutos por causa das árvores.

## Organização

```
data/                  bases brutas (CSV) e o script que as baixa das fontes
notebooks/
  01_eda_rent99.ipynb              análise exploratória do Rent99
  02_regressao_rent99.ipynb        linear, ridge, lasso, polinomial, árvores, bagging e random forest
  03_eda_dry_bean.ipynb            análise exploratória do Dry Bean
  04_classificacao_dry_bean.ipynb  logística, LDA, QDA, KNN com PCA, árvores, bagging e random forest
  05_redes_neurais.ipynb           MLP de classificação (Dry Bean) e de regressão (Rent99)
  utils/                           carregamento, divisão treino/teste, métricas e gráficos de apoio
resultados/            tabelas de métricas no teste, geradas pelos notebooks
relatorio_final.ipynb  síntese: dados, método, comparação dos modelos e conclusão
```

## O que é igual em todos os notebooks

Para que os modelos possam ser comparados, tudo o que não é o modelo fica fixo em `notebooks/utils/dados.py`:

- **Semente**: `SEED = 67` em toda divisão e em todo modelo com sorteio.
- **Rent99**: sai a coluna `rentsqm` (é `rent / area`, então vaza o alvo); 80% para treino e validação (2.465 imóveis) e 20% para o teste final (617).
- **Dry Bean**: saem as 68 linhas duplicadas; divisão estratificada, com 20% de cada classe no teste (10.831 de treino e 2.712 de teste).
- **Pré-processamento** (padronização, PCA, escolha de variáveis) sempre ajustado só no treino e depois aplicado ao teste.
- **Métricas**: RMSE, MAE e R² na regressão; acurácia, acurácia balanceada e F1 macro na classificação, porque as classes do Dry Bean são desbalanceadas.

## Dados

| base | problema | linhas | fonte |
|---|---|---|---|
| Rent99 | regressão | 3.082 | pacote R `gamlss.data`: https://rdrr.io/cran/gamlss.data/man/rent99.html |
| Dry Bean | classificação | 13.611 | UCI: https://archive.ics.uci.edu/dataset/602/dry+bean+dataset |

O dicionário das variáveis está em [data/README.md](data/README.md).

## Resultados

As tabelas com as métricas de todos os modelos no teste ficam em `resultados/` e são reunidas no `relatorio_final.ipynb`. Em resumo:

- **Regressão** (RMSE no teste): polinomial de grau 2 146,3; rede neural 147,7; linear, lasso e ridge 148,8; bagging 151,0; random forest 154,8; árvores únicas 156,0 a 161,6. O R² fica perto de 0,5 em todos, e os erros vêm principalmente dos aluguéis mais altos.
- **Classificação** (acurácia no teste): rede neural 93,1%; KNN 92,5%; logística 91,7%; QDA 91,7%; LDA 91,1%; árvores 89–90%; random forest 75,5%. Os erros se concentram nos pares DERMASON × SIRA e CALI × BARBUNYA.
- As árvores da `rbg` são lentas, então no Dry Bean elas foram treinadas em uma subamostra de 1.500 grãos e com poucas árvores.

## Grupo

Carlos Eduardo Rocha de Paula, Cristiane Magarinos Sampaio, Giovani Firmino Mota, Lisandra Menezes Fagundes, Lucas Fraga Damasceno, Lucca Moulin Cruz e Theo Veiga Drummond Ambrósio.
