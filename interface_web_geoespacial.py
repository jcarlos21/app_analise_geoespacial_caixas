
import streamlit as st
import pandas as pd
from agente_caixas_geoespacial import analisar_distancia_entre_pontos, gerar_mapa_interativo
import tempfile
import os

st.set_page_config(page_title="Análise Geoespacial - Caixas de Emenda", layout="wide")
st.title("📍 Análise Geoespacial de Caixas de Emenda ")
st.markdown("Envie os arquivos com os **pontos de referência** e as **caixas de emenda óptica** para identificar a infraestrutura mais próxima e avaliar viabilidade.")

# Upload dos arquivos
col1, col2 = st.columns(2)
with col1:
    pontos_file = st.file_uploader("📌 Arquivo de Pontos de Referência (Excel)", type=[".xlsx"])
with col2:
    caixas_file = st.file_uploader("🛠️ Arquivo de Caixas de Emenda (Excel)", type=[".xlsx"])

# Configuração de parâmetro
limite = st.slider("Limite de Distância para Viabilidade (m)", 50, 1000, 350, 50)

if pontos_file and caixas_file:
    df_pontos = pd.read_excel(pontos_file)
    df_caixas = pd.read_excel(caixas_file)

    with st.spinner("Calculando distâncias e avaliando viabilidade..."):
        df_resultado = analisar_distancia_entre_pontos(df_pontos, df_caixas, limite)

    st.success("Análise concluída!")
    st.dataframe(df_resultado)

    # Download da tabela
    csv = df_resultado.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Baixar Resultado em CSV", data=csv, file_name="resultado_geoespacial.csv", mime="text/csv")

    # Gerar mapa interativo temporariamente
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        mapa_path = tmp.name
        gerar_mapa_interativo(df_resultado, mapa_path)
        with open(mapa_path, 'r', encoding='utf-8') as f:
            mapa_html = f.read()

    st.components.v1.html(mapa_html, height=600, scrolling=True)

    # Limpar arquivo temporário
    os.remove(mapa_path)
