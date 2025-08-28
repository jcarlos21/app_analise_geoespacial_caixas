import streamlit as st
import pandas as pd
from agente_caixas_geoespacial import (
    analisar_distancia_entre_pontos,
    gerar_mapa_interativo,
    calcular_rota_osrm,
    gerar_rota_por_postes,
)
import tempfile
import os
import folium

st.set_page_config(page_title="Análise Geoespacial - Caixas de Emenda", layout="wide")
st.title("📍 Análise Geoespacial de Caixas de Emenda")
st.markdown("Envie os arquivos ou informe uma localização para identificar a caixa de emenda óptica mais próxima.")

# --- Guia rápido de preparação dos arquivos ---
with st.expander("📚 Como preparar seus arquivos"):
    st.markdown(
        """
**Formato aceito:** arquivos Excel com extensão **`.xlsx`**.

> Dica: Se seu arquivo estiver em outro formato (ex.: `.xls`, `.csv`), abra no Excel/LibreOffice e **exporte** como `.xlsx`.
Não renomeie a extensão manualmente.

### 1) Arquivo de **Caixas de Emenda** (obrigatório)
- **Colunas mínimas obrigatórias**: `Latitude`, `Longitude`
- **Colunas recomendadas**: `Cidade`, `Estado` (ou `UF`), `Pasta`, `Sigla`
- **Tolerância a nomes**: o sistema tenta reconhecer variações comuns (ex.: `lat`, `y`, `lon`, `x`, `município`, `sigla uf`, `identificador`, `id`, `nome`).
- **Exemplo mínimo**:

`SIGLA | LATITUDE | LONGITUDE | CIDADE | ESTADO | PASTA`    
`CE383 | -5.-5.57368002 | -36.92181531 | Natal | RN | Açu - RN`


### 2) **Pontos de Referência**
Você pode fornecer de **duas formas**:

**(A) Enviar arquivo Excel `.xlsx`**  
- Colunas esperadas:
  - Obrigatório: `LATITUDE`, `LONGITUDE`
  - Opcionais: `NOME`, `CIDADE`, `ESTADO`
- Exemplo:

`NOME | LATITUDE | LONGITUDE | CIDADE | ESTADO`     
`Ponto Escola A | -5.63975 | -35.42366 | Natal | RN`

**(B) Informar localização manualmente**  
- Use o formato: `latitude, longitude`  
  - Exemplo: `-5.642754149445223, -35.42481501421498`

### 3) Arquivo de **Postes** (opcional)
- **Quando usar**: marque “Considerar postes para refinar a rota”.
- **Colunas mínimas**: `Latitude`, `Longitude`.
- **Dica**: Inclua o máximo de postes possível no raio de atuação; o buffer (em metros) é ajustado na interface.

### Validações e Mensagens
- Se um `.xlsx` estiver **corrompido** ou não for um Excel real, será exibida uma mensagem para corrigir o arquivo.
- Se colunas obrigatórias estiverem **ausentes**, você verá um aviso indicando quais colunas faltam.
- Se os dados estiverem **vazios**, a análise resultará em uma tabela vazia (sem erro crítico).

Se tiver dúvidas sobre o preparo dos arquivos, me diga como estão seus dados que eu te ajudo a ajustar.
        """
    )

# Arquivo principal de caixas (somente .xlsx permitido pelo Streamlit)
caixas_file = st.file_uploader("🛠️ Arquivo de Caixas de Emenda (Excel)", type=[".xlsx"])
limite = st.slider("Limite de Distância para Viabilidade (m)", 50, 1000, 500, 25)

# === Top-K de caixas a testar ===
k_top = st.slider("Número de caixas a testar (Top-K)", min_value=1, max_value=5, value=3, step=1)
st.caption("⚠️ Use com cautela: quanto maior o Top-K, **mais lento** será o processamento e **maior** será o uso da API de rotas.")

# === POSTES (opcional) ===
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
            st.error("⚠️ Não foi possível ler o arquivo de Postes. Envie um Excel válido (.xlsx). Dica: não renomeie outros formatos para .xlsx.")
            df_postes = None  # segue sem postes

opcao = st.radio("Como deseja fornecer o(s) ponto(s) de referência?", ["📄 Enviar arquivo Excel", "🧭 Informar localização manualmente"])

if caixas_file:
    # Leitura das caixas com tratamento de erro
    try:
        df_caixas = pd.read_excel(caixas_file)
    except Exception:
        st.error("⚠️ Não foi possível ler o arquivo de Caixas. Envie um Excel válido (.xlsx). "
                 "Se o arquivo for de outro formato, exporte/convert a partir do Excel/LibreOffice.")
        st.stop()

    # Filtro por cidade — tolerante à ausência da coluna
    try:
        cidades_disponiveis = sorted(df_caixas['Cidade'].dropna().unique())
        cidades_selecionadas = st.multiselect(
            "Filtrar por cidade das caixas disponíveis:",
            options=cidades_disponiveis,
            default=cidades_disponiveis
        )
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
                # === Elementos de progresso ===
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

            df_pontos = pd.DataFrame([{
                "NOME": "Ponto Manual",
                "LATITUDE": lat,
                "LONGITUDE": lon
            }])

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

    if df_resultado is not None and not df_resultado.empty:
        st.success("Análise concluída!")

        # >>> BLOCO RESTAURADO: botão para baixar o KMZ único
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

        # Exibe tabela (sem a coluna de caminho)
        try:
            df_mostrar = df_resultado.drop(columns=["Download da Rota (KMZ)"])
        except Exception:
            df_mostrar = df_resultado
        st.dataframe(df_mostrar)

# Rodapé
st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'>Desenvolvido por José Carlos dos Santos para a Evos.</div>", unsafe_allow_html=True)
