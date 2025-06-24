
import streamlit as st
import pandas as pd
from agente_caixas_geoespacial import analisar_distancia_entre_pontos, gerar_mapa_interativo
import tempfile
import os

st.set_page_config(page_title="Análise Geoespacial - Caixas de Emenda", layout="wide")
st.title("📍 Análise Geoespacial de Caixas de Emenda")
st.markdown("Envie os arquivos ou informe uma localização para identificar a caixa de emenda óptica mais próxima.")

# Upload do arquivo de caixas
caixas_file = st.file_uploader("🛠️ Arquivo de Caixas de Emenda (Excel)", type=[".xlsx"])
limite = st.slider("Limite de Distância para Viabilidade (m)", 50, 1000, 350, 50)

# Opções de entrada do ponto de referência
opcao = st.radio("Como deseja fornecer o(s) ponto(s) de referência?", ["📄 Enviar arquivo Excel", "🧭 Informar localização manualmente"])

if caixas_file:
    df_caixas = pd.read_excel(caixas_file)

    if opcao == "📄 Enviar arquivo Excel":
        pontos_file = st.file_uploader("📌 Arquivo de Pontos de Referência (Excel)", type=[".xlsx"])
        if pontos_file:
            df_pontos = pd.read_excel(pontos_file)
            with st.spinner("Calculando distâncias e avaliando viabilidade..."):
                df_resultado = analisar_distancia_entre_pontos(df_pontos, df_caixas, limite)

    else:
        with st.form("form_coords"):
            nome = st.text_input("Nome do ponto de referência", "Ponto Manual")
            cidade = st.text_input("Cidade", "Exemplo")
            estado = st.text_input("Estado", "XX")
            lat = st.number_input("Latitude", format="%.8f")
            lon = st.number_input("Longitude", format="%.8f")
            submitted = st.form_submit_button("Calcular")

        if submitted:
            df_pontos = pd.DataFrame([{
                "Nome": nome,
                "Cidade": cidade,
                "Estado": estado,
                "LATITUDE": lat,
                "LONGITUDE": lon
            }])
            with st.spinner("Calculando distância para o ponto informado..."):
                df_resultado = analisar_distancia_entre_pontos(df_pontos, df_caixas, limite)

    if 'df_resultado' in locals():
        st.success("Análise concluída!")
        st.dataframe(df_resultado)

        csv = df_resultado.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Baixar Resultado em CSV", data=csv, file_name="resultado_geoespacial.csv", mime="text/csv")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
            mapa_path = tmp.name
            gerar_mapa_interativo(df_resultado, mapa_path)
            with open(mapa_path, 'r', encoding='utf-8') as f:
                mapa_html = f.read()

        st.components.v1.html(mapa_html, height=600, scrolling=True)
        os.remove(mapa_path)
