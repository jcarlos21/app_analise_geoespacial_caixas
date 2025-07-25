import streamlit as st
import pandas as pd
from agente_caixas_geoespacial import analisar_distancia_entre_pontos, gerar_mapa_interativo
import tempfile
import os

st.set_page_config(page_title="Análise Geoespacial - Caixas de Emenda", layout="wide")
st.title("📍 Análise Geoespacial de Caixas de Emenda")
st.markdown("Envie os arquivos ou informe uma localização para identificar a caixa de emenda óptica mais próxima.")

caixas_file = st.file_uploader("🛠️ Arquivo de Caixas de Emenda (Excel)", type=[".xlsx"])
limite = st.slider("Limite de Distância para Viabilidade (m)", 50, 1000, 350, 50)

opcao = st.radio("Como deseja fornecer o(s) ponto(s) de referência?", ["📄 Enviar arquivo Excel", "🧭 Informar localização manualmente"])

if caixas_file:
    df_caixas = pd.read_excel(caixas_file)

    # Filtro por cidade
    cidades_disponiveis = sorted(df_caixas['Cidade'].dropna().unique())
    cidades_selecionadas = st.multiselect(
        "Filtrar por cidade das caixas disponíveis:",
        options=cidades_disponiveis,
        default=cidades_disponiveis
    )
    df_caixas_filtrado = df_caixas[df_caixas['Cidade'].isin(cidades_selecionadas)]

    if opcao == "📄 Enviar arquivo Excel":
        pontos_file = st.file_uploader("📌 Arquivo de Pontos de Referência (Excel)", type=[".xlsx"])
        if pontos_file:
            df_pontos = pd.read_excel(pontos_file)
            with st.spinner("Calculando distâncias e avaliando viabilidade..."):
                df_resultado = analisar_distancia_entre_pontos(df_pontos, df_caixas_filtrado, limite)

    else:
        with st.form("form_coords"):
            localizacao_str = st.text_input("Localização (formato: latitude, longitude)", "-5.642754149445223, -35.42481501421498")
            submitted = st.form_submit_button("Calcular")

        if submitted:
            try:
                lat_str, lon_str = [x.strip() for x in localizacao_str.split(",")]
                lat = float(lat_str)
                lon = float(lon_str)
                df_pontos = pd.DataFrame([{
                    "NOME": "Ponto Manual",
                    "CIDADE": "",
                    "ESTADO": "",
                    "LATITUDE": lat,
                    "LONGITUDE": lon
                }])
                with st.spinner("Calculando distância para o ponto informado..."):
                    df_resultado = analisar_distancia_entre_pontos(df_pontos, df_caixas_filtrado, limite)
            except Exception as e:
                st.error(f"Erro ao interpretar a localização: {e}")
                df_resultado = None

    if 'df_resultado' in locals() and df_resultado is not None:
        st.success("Análise concluída!")

        # Criar botão único de download, se o KMZ existir
        kmz_path_unico = df_resultado["Download da Rota (KMZ)"].iloc[0]
        if os.path.exists(kmz_path_unico):
            with open(kmz_path_unico, "rb") as f:
                st.download_button("📥 Baixar KMZ com todas as rotas", f.read(), file_name=os.path.basename(kmz_path_unico), mime="application/vnd.google-earth.kmz")

        df_mostrar = df_resultado.drop(columns=["Download da Rota (KMZ)"])
        st.dataframe(df_mostrar)

        # Gerar e exibir o mapa
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
            mapa_path = tmp.name
            gerar_mapa_interativo(df_resultado, mapa_path)
            with open(mapa_path, 'r', encoding='utf-8') as f:
                mapa_html = f.read()

        st.components.v1.html(mapa_html, height=600, scrolling=True)
        os.remove(mapa_path)

# Rodapé
st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'>Desenvolvido por José Carlos dos Santos</div>", unsafe_allow_html=True)