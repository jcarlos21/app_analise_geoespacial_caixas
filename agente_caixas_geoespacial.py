import pandas as pd
import requests
from geopy.distance import geodesic
import folium
from folium.plugins import MarkerCluster
import simplekml
import os

def calcular_rota_osrm(coord_origem, coord_destino):
    lat1, lon1 = coord_origem
    lat2, lon2 = coord_destino

    url = (
        f"https://router.project-osrm.org/route/v1/driving/"
        f"{lon1},{lat1};{lon2},{lat2}"
        f"?overview=full&geometries=geojson"
    )

    try:
        resp = requests.get(url)
        data = resp.json()
        rota_coords = data['routes'][0]['geometry']['coordinates']
        distancia_metros = data['routes'][0]['distance']
        return rota_coords, distancia_metros
    except Exception as e:
        print(f"Erro ao calcular rota OSRM: {e}")
        return [], 0

def gerar_kmz(nome_base, rota_coords):
    import os
    pasta_kmz = os.path.join(os.getcwd(), "rotas_kmz")
    os.makedirs(pasta_kmz, exist_ok=True)  # cria a pasta se não existir

    kml = simplekml.Kml()
    ls = kml.newlinestring(name=nome_base)
    ls.coords = [(lon, lat) for lon, lat in rota_coords]
    ls.style.linestyle.width = 3
    ls.style.linestyle.color = simplekml.Color.red

    kml_path = os.path.join(pasta_kmz, f"{nome_base}.kmz")
    kml.savekmz(kml_path)
    return kml_path

def analisar_distancia_entre_pontos(df_pontos, df_caixas, limite_fibra=350):
    df_pontos.columns = df_pontos.columns.str.strip().str.upper()
    df_pontos.rename(columns={
        'NOME': 'Nome',
        'NOME MUNICÍPIO': 'Cidade',
        'SIGLA UF': 'Estado',
        'LATITUDE': 'LATITUDE',
        'LONGITUDE': 'LONGITUDE'
    }, inplace=True)

    resultados = []

    for index, ponto in df_pontos.iterrows():
        coord_ponto = (ponto['LATITUDE'], ponto['LONGITUDE'])

        menor_dist_geodesica = float('inf')
        caixa_proxima = None

        for _, caixa in df_caixas.iterrows():
            coord_caixa = (caixa['Latitude'], caixa['Longitude'])
            dist_geo = geodesic(coord_ponto, coord_caixa).meters
            if dist_geo < menor_dist_geodesica:
                menor_dist_geodesica = dist_geo
                caixa_proxima = caixa

        coord_caixa_proxima = (caixa_proxima['Latitude'], caixa_proxima['Longitude'])
        rota_coords, distancia_real = calcular_rota_osrm(coord_ponto, coord_caixa_proxima)
        if not rota_coords:
            distancia_real = menor_dist_geodesica

        nome_base = f"rota_ponto_{index+1}"
        kmz_path = gerar_kmz(nome_base, rota_coords) if rota_coords else ""

        resultados.append({
            'Nome do Ponto de Referência': ponto.get('Nome', ''),
            'Cidade do Ponto': ponto.get('Cidade', ''),
            'Estado do Ponto': ponto.get('Estado', ''),
            'Localização do Ponto': f"{ponto['LATITUDE']}, {ponto['LONGITUDE']}",
            'Localização da Caixa': f"{caixa_proxima['Latitude']}, {caixa_proxima['Longitude']}",
            'Identificador': caixa_proxima['Sigla'],
            'Cidade': caixa_proxima['Cidade'],
            'Estado': caixa_proxima['Estado'],
            'Categoria': caixa_proxima['Pasta'],
            'Distância da Rota (m)': round(distancia_real, 2),
            'Viabilidade': 'Conectável' if distancia_real < limite_fibra else '',
            'Download da Rota (KMZ)': kmz_path
        })

    return pd.DataFrame(resultados)

def gerar_mapa_interativo(df_resultados, caminho_html):
    mapa = folium.Map(location=[-5.8, -36.6], zoom_start=8)
    marcadores = MarkerCluster().add_to(mapa)

    for _, linha in df_resultados.iterrows():
        lat_ponto, lon_ponto = map(float, linha['Localização do Ponto'].split(', '))
        lat_caixa, lon_caixa = map(float, linha['Localização da Caixa'].split(', '))

        rota_coords, _ = calcular_rota_osrm((lat_ponto, lon_ponto), (lat_caixa, lon_caixa))
        if rota_coords:
            rota_convertida = [(lat, lon) for lon, lat in rota_coords]
            folium.PolyLine(
                locations=rota_convertida,
                color='red', weight=3, tooltip="Rota real"
            ).add_to(mapa)
        else:
            folium.PolyLine(
                locations=[[lat_ponto, lon_ponto], [lat_caixa, lon_caixa]],
                color='gray', weight=2, dash_array="5,5", tooltip="Rota linear (falha OSRM)"
            ).add_to(mapa)

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

    mapa.save(caminho_html)
    return caminho_html