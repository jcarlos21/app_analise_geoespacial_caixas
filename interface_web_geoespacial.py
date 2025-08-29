import streamlit as st 
import pandas as pd
from agente_caixas_geoespacial import (
    analisar_distancia_entre_pontos,
    gerar_mapa_interativo,
    calcular_rota_osrm,
    gerar_rota_por_postes,
    gerar_descritivos_zip,
)
import os
import folium

st.set_page_config(page_title="Análise Geoespacial - Caixas de Emenda", layout="wide")
st.title("📍 Análise Geoespacial de Caixas de Emenda")
st.markdown("Envie os arquivos ou informe uma localização para identificar a caixa de emenda óptica mais próxima.")

# Estado persistente para manter resultados entre reruns
if "df_resultado" not in st.session_state:
    st.session_state["df_resultado"] = None

with st.expander("📚 Como preparar seus arquivos"):
    st.markdown(
        """
**Formato aceito:** arquivos Excel com extensão **`.xlsx`**.

> Dica: Se seu arquivo estiver em outro formato (ex.: `.xls`, `.csv`), abra no Excel/LibreOffice e **exporte** como `.xlsx`.
Não renomeie a extensão manualmente.
...
        """
    )

caixas_file = st.file_uploader("🛠️ Arquivo de Caixas de Emenda (Excel)", type=[".xlsx"])
limite = st.slider("Limite de Distância para Viabilidade (m)", 50, 1000, 500, 25)

k_top = st.slider("Número de caixas a testar (Top-K)", min_value=1, max_value=5, value=3, step=1)
st.caption("⚠️ Quanto maior o Top-K, mais lento e mais uso de API de rotas.")

st.markdown("### 🌲 Postes (opcional)")
usar_postes = st.checkbox("Considerar postes para refinar a rota", value=False)
buffer_postes_m = None
df_postes = None
if usar_postes:
    buffer_postes_m = st.slider("Largura do buffer ao redor da rota (m)", 5, 50, 20, 1)
    postes_file = st.file_uploader("📎 Arquivo de Postes (Excel)", type=[".xlsx"])
    if postes_file:
        try:
            df_postes = pd.read_excel(postes_file)
        except Exception:
            st.error("⚠️ Não foi possível ler o arquivo de Postes. Envie um Excel válido (.xlsx).")
            df_postes = None

opcao = st.radio("Como deseja fornecer o(s) ponto(s) de referência?", ["📄 Enviar arquivo Excel", "🧭 Informar localização manualmente"])

if caixas_file:
    try:
        df_caixas = pd.read_excel(caixas_file)
    except Exception:
        st.error("⚠️ Não foi possível ler o arquivo de Caixas. Envie um Excel válido (.xlsx).")
        st.stop()

    try:
        cidades_disponiveis = sorted(df_caixas['Cidade'].dropna().unique())
        cidades_selecionadas = st.multiselect("Filtrar por cidade das caixas disponíveis:", options=cidades_disponiveis, default=cidades_disponiveis)
        df_caixas_filtrado = df_caixas[df_caixas['Cidade'].isin(cidades_selecionadas)]
    except Exception:
        st.info("ℹ️ Não encontrei a coluna **'Cidade'** para filtrar. A análise continuará sem filtro de cidade.")
        df_caixas_filtrado = df_caixas

    df_resultado = None

    if opcao == "📄 Enviar arquivo Excel":
        pontos_file = st.file_uploader("📌 Arquivo de Pontos de Referência (Excel)", type=[".xlsx"])
        if pontos_file:
            try:
                df_pontos = pd.read_excel(pontos_file)
            except Exception:
                st.error("⚠️ Não foi possível ler o arquivo de Pontos. Envie um Excel válido (.xlsx).")
                df_pontos = None

            if df_pontos is not None:
                progresso_text = st.empty()
                progresso_bar = st.progress(0)

                def _progress_cb(done, total):
                    pct = int((done / max(1, total)) * 100)
                    progresso_text.markdown(f"Calculando distâncias e avaliando viabilidade... **{done}/{total}** pontos.")
                    progresso_bar.progress(min(max(pct, 0), 100))

                with st.spinner("Processando rotas..."):
                    try:
                        df_resultado = analisar_distancia_entre_pontos(
                            df_pontos, df_caixas_filtrado, limite,
                            df_postes=df_postes, buffer_postes_m=buffer_postes_m,
                            usar_postes=usar_postes, k_top=k_top,
                            progress_cb=_progress_cb
                        )
                    except Exception as e:
                        st.error(f"❌ Ocorreu um erro inesperado: {e}")
                        df_resultado = None

                if df_resultado is not None:
                    progresso_text.markdown(f"Análise concluída! **{len(df_pontos)}/{len(df_pontos)}** pontos.")
                    progresso_bar.progress(100)
                    st.session_state["df_resultado"] = df_resultado

    else:
        with st.form("form_coords"):
            localizacao_str = st.text_input("Localização (formato: latitude, longitude)", "-5.642754149445223, -35.42481501421498")
            submitted = st.form_submit_button("Calcular")

        if submitted:
            try:
                lat_str, lon_str = [x.strip() for x in localizacao_str.split(",")]
                lat = float(lat_str); lon = float(lon_str)
            except Exception:
                st.error("⚠️ Formato inválido. Use: latitude, longitude")
                st.stop()

            df_pontos = pd.DataFrame([{"NOME": "Ponto Manual", "LATITUDE": lat, "LONGITUDE": lon}])

            progresso_text = st.empty()
            progresso_bar = st.progress(0)

            def _progress_cb(done, total):
                pct = int((done / max(1, total)) * 100)
                progresso_text.markdown(f"Calculando distâncias e avaliando viabilidade... **{done}/{total}** pontos.")
                progresso_bar.progress(min(max(pct, 0), 100))

            with st.spinner("Processando rota..."):
                try:
                    df_resultado = analisar_distancia_entre_pontos(
                        df_pontos, df_caixas_filtrado, limite,
                        df_postes=df_postes, buffer_postes_m=buffer_postes_m,
                        usar_postes=usar_postes, k_top=k_top,
                        progress_cb=_progress_cb
                    )
                except Exception as e:
                    st.error(f"❌ Ocorreu um erro inesperado: {e}")
                    df_resultado = None

            progresso_text.markdown("Análise concluída! **1/1** pontos.")
            progresso_bar.progress(100)
            st.session_state["df_resultado"] = df_resultado

# =========================
# Renderização dos resultados
# =========================
df_resultado = st.session_state.get("df_resultado")
if df_resultado is not None and not df_resultado.empty:
    st.success("Análise concluída!")

    kmz_path_unico = None
    try:
        if "Download da Rota (KMZ)" in df_resultado.columns and len(df_resultado) > 0:
            kmz_path_unico = df_resultado["Download da Rota (KMZ)"].iloc[0]
    except Exception:
        kmz_path_unico = None

    if isinstance(kmz_path_unico, str) and os.path.exists(kmz_path_unico):
        try:
            with open(kmz_path_unico, "rb") as f:
                st.download_button(
                    "📥 Baixar KMZ com todas as rotas",
                    f.read(),
                    file_name=os.path.basename(kmz_path_unico),
                    mime="application/vnd.google-earth.kmz"
                )
        except Exception:
            st.warning("⚠️ Não foi possível preparar o download do KMZ no momento.")

    try:
        df_mostrar = df_resultado.drop(columns=["Download da Rota (KMZ)"])
    except Exception:
        df_mostrar = df_resultado
    st.dataframe(df_mostrar)

    st.markdown("### 🧾 Descritivos por ponto (DOCX + Print Satelital)")
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        rota_no_print = st.selectbox("Rotas no descritivo", options=["OSRM", "Postes", "Ambas"], index=2)
    with col2:
        flag_postes_pts = st.checkbox("Incluir marcadores dos postes (pontos)", value=False)
    with col3:
        st.caption("O print usa mapa satelital (Esri.WorldImagery).")

    if st.button("Gerar descritivos (.zip)"):
        with st.spinner("Gerando descritivos..."):
            try:
                zip_path = gerar_descritivos_zip(
                    df_resultado,
                    incluir_rota_postes=(rota_no_print in ["Postes", "Ambas"]),
                    incluir_postes_pts=flag_postes_pts,
                    rota_no_print=rota_no_print
                )
                with open(zip_path, "rb") as f:
                    zip_bytes = f.read()
                st.success("Descritivos gerados com sucesso!")
                st.download_button(
                    "⬇️ Baixar descritivos (.zip)",
                    data=zip_bytes,
                    file_name="descritivos_por_ponto.zip",
                    mime="application/zip"
                )
            except Exception as e:
                st.error(f"❌ Falha ao gerar descritivos: {e}")

st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'>Desenvolvido por José Carlos dos Santos para a Evos.</div>", unsafe_allow_html=True)
