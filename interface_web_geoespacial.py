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
            # Leitura dos pontos com tratamento
            try:
                df_pontos = pd.read_excel(pontos_file)
            except Exception:
                st.error("⚠️ Não foi possível ler o arquivo de Pontos. Envie um Excel válido (.xlsx).")
                df_pontos = None

            if df_pontos is not None:
                with st.spinner("Calculando distâncias e avaliando viabilidade..."):
                    try:
                        df_resultado = analisar_distancia_entre_pontos(
                            df_pontos, df_caixas_filtrado, limite,
                            df_postes=df_postes, buffer_postes_m=buffer_postes_m, usar_postes=usar_postes
                        )
                    except KeyError as e:
                        # Falhas de colunas obrigatórias são levantadas pelo _normalize_caixas_df
                        st.error(f"⚠️ Problema de colunas obrigatórias no arquivo: {e}. "
                                 "Verifique se a planilha de **Caixas** possui Latitude/Longitude (nomes semelhantes são aceitos).")
                        df_resultado = None
                    except Exception as e:
                        st.error(f"❌ Ocorreu um erro inesperado durante a análise: {e}")
                        df_resultado = None

    else:
        with st.form("form_coords"):
            localizacao_str = st.text_input("Localização (formato: latitude, longitude)", "-5.642754149445223, -35.42481501421498")
            submitted = st.form_submit_button("Calcular")

        if submitted:
            # Interpretação das coordenadas fornecidas manualmente
            try:
                lat_str, lon_str = [x.strip() for x in localizacao_str.split(",")]
                lat = float(lat_str); lon = float(lon_str)
            except Exception:
                st.error("⚠️ Formato inválido. Use o padrão: **latitude, longitude** (ex.: `-5.6427541494, -35.4248150142`).")
                st.stop()

            df_pontos = pd.DataFrame([{
                "NOME": "Ponto Manual",
                "CIDADE": "",
                "ESTADO": "",
                "LATITUDE": lat,
                "LONGITUDE": lon
            }])

            with st.spinner("Calculando distância para o ponto informado..."):
                try:
                    df_resultado = analisar_distancia_entre_pontos(
                        df_pontos, df_caixas_filtrado, limite,
                        df_postes=df_postes, buffer_postes_m=buffer_postes_m, usar_postes=usar_postes
                    )
                except KeyError as e:
                    st.error(f"⚠️ Problema de colunas obrigatórias no arquivo de Caixas: {e}. "
                             "Verifique se a planilha possui Latitude/Longitude (nomes semelhantes são aceitos).")
                    df_resultado = None
                except Exception as e:
                    st.error(f"❌ Ocorreu um erro inesperado durante a análise: {e}")
                    df_resultado = None

    if df_resultado is not None:
        if df_resultado.empty:
            st.warning("Nenhum resultado foi gerado. Verifique se os dados enviados possuem linhas válidas.")
        else:
            st.success("Análise concluída!")

            # Botão de download do KMZ único — seguro contra ausências
            kmz_path_unico = None
            try:
                if "Download da Rota (KMZ)" in df_resultado.columns and len(df_resultado) > 0:
                    kmz_path_unico = df_resultado["Download da Rota (KMZ)"].iloc[0]
            except Exception:
                kmz_path_unico = None

            if isinstance(kmz_path_unico, str) and os.path.exists(kmz_path_unico):
                try:
                    with open(kmz_path_unico, "rb") as f:
                        st.download_button("📥 Baixar KMZ com todas as rotas", f.read(),
                                           file_name=os.path.basename(kmz_path_unico),
                                           mime="application/vnd.google-earth.kmz")
                except Exception:
                    st.warning("⚠️ Não foi possível preparar o download do KMZ no momento.")

            # Exibe tabela (sem a coluna de caminho)
            try:
                df_mostrar = df_resultado.drop(columns=["Download da Rota (KMZ)"])
            except Exception:
                df_mostrar = df_resultado

            st.dataframe(df_mostrar)

            # === MAPA: padrão + sobreposição opcional da rota por postes ===
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
                    mapa_path = tmp.name

                    if not (usar_postes and df_postes is not None and buffer_postes_m):
                        # caminho padrão com tratamento
                        try:
                            gerar_mapa_interativo(df_resultado, mapa_path)
                        except Exception as e:
                            st.warning(f"⚠️ Não foi possível gerar o mapa interativo padrão: {e}")
                    else:
                        # desenho detalhado com postes e tratamento
                        m = folium.Map(location=[-5.8, -36.6], zoom_start=8)
                        for _, linha in df_resultado.iterrows():
                            try:
                                lat_ponto, lon_ponto = map(float, linha['Localização do Ponto'].split(', '))
                                lat_caixa, lon_caixa = map(float, linha['Localização da Caixa'].split(', '))
                            except Exception:
                                st.warning("⚠️ Não foi possível interpretar coordenadas de uma das linhas de resultado.")
                                continue

                            try:
                                rota_coords_osrm, _ = calcular_rota_osrm((lat_caixa, lon_caixa), (lat_ponto, lon_ponto))
                                if rota_coords_osrm:
                                    rota_convertida = [(lat, lon) for lon, lat in rota_coords_osrm]
                                    folium.PolyLine(
                                        locations=rota_convertida,
                                        color='red', weight=4,
                                        tooltip=f"Rota OSRM: {linha.get('Distância da Rota (m)','?')} metros."
                                    ).add_to(m)
                                    try:
                                        rota_postes, dist_postes = gerar_rota_por_postes(
                                            rota_coords_osrm, df_postes, buffer_m=float(buffer_postes_m),
                                            coord_caixa=(lat_caixa, lon_caixa), coord_ponto=(lat_ponto, lon_ponto)
                                        )
                                        if rota_postes:
                                            rota_postes_convertida = [(lat, lon) for lon, lat in rota_postes]
                                            folium.PolyLine(
                                                locations=rota_postes_convertida, color='blue', weight=4,
                                                tooltip=f"Rota Via Postes: {round(dist_postes)} metros"
                                            ).add_to(m)
                                    except Exception as e:
                                        st.warning(f"Falha ao desenhar rota por postes para {linha.get('Nome do Ponto de Referência','')}: {e}")
                                else:
                                    folium.PolyLine(
                                        locations=[[lat_ponto, lon_ponto], [lat_caixa, lon_caixa]],
                                        weight=2, dash_array="5,5", tooltip="Rota linear (falha OSRM)"
                                    ).add_to(m)

                                folium.Marker([lat_ponto, lon_ponto], tooltip=f"{linha.get('Nome do Ponto de Referência','')}", icon=folium.Icon(color='blue', icon='info-sign')).add_to(m)
                                folium.Marker([lat_caixa, lon_caixa], tooltip=f"{linha.get('Identificador','')}", icon=folium.Icon(color='green', icon='hdd', prefix='fa')).add_to(m)
                            except Exception as e:
                                st.warning(f"⚠️ Não foi possível desenhar um dos trechos no mapa: {e}")

                        m.save(mapa_path)

                    with open(mapa_path, 'r', encoding='utf-8') as f:
                        mapa_html = f.read()

                st.components.v1.html(mapa_html, height=600, scrolling=True)
                os.remove(mapa_path)
            except Exception as e:
                st.warning(f"⚠️ Ocorreu um problema ao preparar o mapa interativo: {e}")

# Rodapé
st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'>Desenvolvido por José Carlos dos Santos para a Evos.</div>", unsafe_allow_html=True)
