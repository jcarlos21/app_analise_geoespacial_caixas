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

caixas_file = st.file_uploader("🛠️ Arquivo de Caixas de Emenda (Excel)", type=[".xlsx"])
limite = st.slider("Limite de Distância para Viabilidade (m)", 50, 1000, 350, 50)

# === POSTES (opcional) ===
st.markdown("### 🌲 Postes (opcional)")
usar_postes = st.checkbox("Considerar postes para refinar a rota", value=False)
buffer_postes_m = None
df_postes = None
if usar_postes:
    buffer_postes_m = st.slider("Largura do buffer ao redor da rota (m)", 5, 50, 20, 1)
    postes_file = st.file_uploader("📎 Arquivo de Postes (Excel)", type=[".xlsx"])
    if postes_file:
        df_postes = pd.read_excel(postes_file)

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
                df_resultado = analisar_distancia_entre_pontos(
                    df_pontos, df_caixas_filtrado, limite,
                    df_postes=df_postes, buffer_postes_m=buffer_postes_m, usar_postes=usar_postes
                )

    else:
        with st.form("form_coords"):
            localizacao_str = st.text_input("Localização (formato: latitude, longitude)", "-5.642754149445223, -35.42481501421498")
            submitted = st.form_submit_button("Calcular")

        if submitted:
            try:
                lat_str, lon_str = [x.strip() for x in localizacao_str.split(",")]
                lat = float(lat_str); lon = float(lon_str)
                df_pontos = pd.DataFrame([{
                    "NOME": "Ponto Manual",
                    "CIDADE": "",
                    "ESTADO": "",
                    "LATITUDE": lat,
                    "LONGITUDE": lon
                }])
                with st.spinner("Calculando distância para o ponto informado..."):
                    df_resultado = analisar_distancia_entre_pontos(
                        df_pontos, df_caixas_filtrado, limite,
                        df_postes=df_postes, buffer_postes_m=buffer_postes_m, usar_postes=usar_postes
                    )
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

        # === MAPA: padrão + sobreposição opcional da rota por postes ===
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
            mapa_path = tmp.name

            # Se NÃO for usar postes, mantém comportamento original
            if not (usar_postes and df_postes is not None and buffer_postes_m):
                gerar_mapa_interativo(df_resultado, mapa_path)
            else:
                # Reconstrói um mapa customizado para poder sobrepor a rota por postes
                m = folium.Map(location=[-5.8, -36.6], zoom_start=8)
                for _, linha in df_resultado.iterrows():
                    lat_ponto, lon_ponto = map(float, linha['Localização do Ponto'].split(', '))
                    lat_caixa, lon_caixa = map(float, linha['Localização da Caixa'].split(', '))

                    # Rota OSRM
                    rota_coords_osrm, _ = calcular_rota_osrm((lat_caixa, lon_caixa), (lat_ponto, lon_ponto))
                    if rota_coords_osrm:
                        rota_convertida = [(lat, lon) for lon, lat in rota_coords_osrm]
                        folium.PolyLine(locations=rota_convertida, color='red', weight=4, tooltip=f"Rota OSRM: {linha['Distância da Rota (m)']} metros.").add_to(m)

                        # Rota por Postes (tracejada)
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

                    # Marcadores
                    folium.Marker([lat_ponto, lon_ponto], tooltip=f"{linha['Nome do Ponto de Referência']}", icon=folium.Icon(color='blue', icon='info-sign')).add_to(m)
                    folium.Marker([lat_caixa, lon_caixa], tooltip=f"{linha['Identificador']}", icon=folium.Icon(color='green', icon='hdd', prefix='fa')).add_to(m)

                m.save(mapa_path)

            with open(mapa_path, 'r', encoding='utf-8') as f:
                mapa_html = f.read()

        st.components.v1.html(mapa_html, height=600, scrolling=True)
        os.remove(mapa_path)

# Rodapé
st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'>Desenvolvido por José Carlos dos Santos</div>", unsafe_allow_html=True)
