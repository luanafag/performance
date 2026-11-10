# Dados

Os dois CSVs desta pasta são as bases brutas, sem nenhum tratamento. Toda a limpeza é feita em `notebooks/utils/dados.py`, para que todos os notebooks partam do mesmo ponto.

Para regenerar os arquivos a partir das fontes originais:

```bash
python data/baixar_dados.py
```

## rent99.csv

Pesquisa de aluguéis de Munique de 1999, com 3.082 imóveis. Vem do pacote R `gamlss.data` (CRAN).
Fonte: https://rdrr.io/cran/gamlss.data/man/rent99.html

| coluna | significado |
|---|---|
| `rent` | aluguel líquido mensal, em euros (variável alvo) |
| `rentsqm` | aluguel por metro quadrado (`rent / area`); removida na modelagem por vazar o alvo |
| `area` | área do imóvel em m² |
| `yearc` | ano de construção |
| `location` | qualidade da localização: 1 média, 2 boa, 3 excelente |
| `bath` | banheiro: 0 padrão, 1 premium |
| `kitchen` | cozinha: 0 padrão, 1 premium |
| `cheating` | aquecimento central: 0 não, 1 sim |
| `district` | código do distrito de Munique |

## dry_bean.csv

13.611 grãos de sete tipos de feijão, fotografados e descritos por 16 medidas de tamanho e forma. Vem do UCI Machine Learning Repository (id 602).
Fonte: https://archive.ics.uci.edu/dataset/602/dry+bean+dataset

A coluna alvo é `Class` (Seker, Barbunya, Bombay, Cali, Dermason, Horoz e Sira). No arquivo original duas colunas vêm com o nome `AspectRation` e `roundness`; elas são renomeadas para `AspectRatio` e `Roundness` no carregamento. Há 68 linhas duplicadas, removidas antes da modelagem.
