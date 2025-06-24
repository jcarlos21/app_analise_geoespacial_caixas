
# Agente ChatGPT: Análise Geoespacial de Caixas de Emenda óptica

def analisar_distancia_entre_pontos(df_pontos, df_caixas, limite_fibra=350):
    # Padroniza os nomes das colunas para garantir compatibilidade
    df_pontos.columns = df_pontos.columns.str.strip().str.upper()
    df_pontos.rename(columns={
        'NOME': 'Nome',
        'NOME MUNICÍPIO': 'Cidade',
        'SIGLA UF': 'Estado',
        'LATITUDE': 'LATITUDE',
        'LONGITUDE': 'LONGITUDE'
    }, inplace=True)

    from geopy.distance import geodesic
    import pandas as pd

    resultados = []
    caixas_coords = df_caixas[['Latitude', 'Longitude']].to_numpy()

    for _, ponto in df_pontos.iterrows():
        coord_ponto = (ponto['LATITUDE'], ponto['LONGITUDE'])
        menor_dist = float('inf')
        caixa_proxima = None

        for _, caixa in df_caixas.iterrows():
            coord_caixa = (caixa['Latitude'], caixa['Longitude'])
            dist = geodesic(coord_ponto, coord_caixa).meters
            if dist < menor_dist:
                menor_dist = dist
                caixa_proxima = caixa

        resultados.append({
            'Nome do Ponto de Referência': ponto['Nome'],
            'Cidade do Ponto': ponto['Cidade'],
            'Estado do Ponto': ponto['Estado'],
            'Localização do Ponto': f"{ponto['LATITUDE']}, {ponto['LONGITUDE']}",
            'Localização da Caixa': f"{caixa_proxima['Latitude']}, {caixa_proxima['Longitude']}",
            'Identificador': caixa_proxima['Sigla'],
            'Cidade': caixa_proxima['Cidade'],
            'Estado': caixa_proxima['Estado'],
            'Categoria': caixa_proxima['Pasta'],
            'Distância Linear (m)': round(menor_dist, 2),
            'Viabilidade': 'Conectável' if menor_dist < limite_fibra else ''
        })

    return pd.DataFrame(resultados)

def gerar_mapa_interativo(df_resultados, caminho_html):
    import folium
    from folium.plugins import MarkerCluster

    mapa = folium.Map(location=[-5.8, -36.6], zoom_start=8)
    marcadores = MarkerCluster().add_to(mapa)

    for _, linha in df_resultados.iterrows():
        lat_ponto, lon_ponto = map(float, linha['Localização do Ponto'].split(', '))
        lat_caixa, lon_caixa = map(float, linha['Localização da Caixa'].split(', '))

        folium.Marker(
            location=[lat_ponto, lon_ponto],
            popup=f"<b>Ponto:</b> {linha['Nome do Ponto de Referência']}<br><b>Viabilidade:</b> {linha['Viabilidade'] or 'N/A'}",
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(marcadores)

        folium.Marker(
            location=[lat_caixa, lon_caixa],
            popup=f"<b>Caixa:</b> {linha['Identificador']}<br><b>{linha['Cidade']} - {linha['Estado']}</b>",
            icon=folium.Icon(color='green', icon='hdd', prefix='fa')
        ).add_to(marcadores)

        folium.PolyLine(
            locations=[[lat_ponto, lon_ponto], [lat_caixa, lon_caixa]],
            color='red', weight=2, tooltip=f"{linha['Distância Linear (m)']} m"
        ).add_to(mapa)

    mapa.save(caminho_html)
    return caminho_html
