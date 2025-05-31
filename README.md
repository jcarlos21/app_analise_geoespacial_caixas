
# 🛰️ Análise Geoespacial de Caixas de Emenda Óptica

Esta aplicação interativa permite identificar, de forma rápida e visual, a **caixa de emenda óptica mais próxima** de um ou mais pontos de referência geográficos. Ideal para provedores de internet e equipes de infraestrutura que desejam avaliar a **viabilidade de atendimento via fibra óptica** com base em localização.

## 🚀 Funcionalidades

- 📍 **Entrada flexível**:
  - Upload de planilha Excel com múltiplos pontos
  - Inserção manual de coordenadas (`latitude, longitude`)

- 📦 Upload de arquivo com **caixas de emenda óptica** (Excel)

- 📏 Cálculo da **distância geodésica (reta)** entre ponto e caixas

- ✅ Classificação de viabilidade ("Conectável") com base em distância (limite ajustável)

- 🗺️ **Mapa interativo** com:
  - Marcadores personalizados (ponto e caixa)
  - Linhas conectando pares
  - Tooltips com distância e viabilidade

- 📤 Exportação de resultados em **CSV**

## 🛠️ Tecnologias Utilizadas

- [Streamlit](https://streamlit.io/) – Interface web interativa
- [Pandas](https://pandas.pydata.org/) – Manipulação de dados
- [Geopy](https://geopy.readthedocs.io/) – Cálculo de distância geodésica
- [Folium](https://python-visualization.github.io/folium/) – Mapa interativo baseado em Leaflet.js

## 📦 Como usar

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Execute a aplicação:
```bash
streamlit run interface_web_geoespacial.py
```

## 📁 Estrutura Esperada dos Arquivos

### Arquivo de pontos (Excel)
Deve conter colunas:
- `Nome`, `LATITUDE`, `LONGITUDE`, `Cidade`, `Estado`

### Arquivo de caixas de emenda (Excel)
Deve conter colunas:
- `Latitude`, `Longitude`, `Sigla`, `Cidade`, `Estado`, `Pasta`

## 🧭 Exemplo de coordenada manual
```
-5.642754149445223, -35.42481501421498
```

## 📃 Licença
Este projeto está licenciado sob a licença MIT.
