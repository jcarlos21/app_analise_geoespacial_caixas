import pandas as pd
import requests
from geopy.distance import geodesic
import folium
from folium.plugins import MarkerCluster
import simplekml
import os

def calcular_rota_osrm(coord_origem, coord_destino):
    lat1, lon1 = coord_origem  # caixa
    lat2, lon2 = coord_destino  # ponto consultado

    url = (
        f"https://router.project-osrm.org/route/v1/driving/"
        f"{lon1},{lat1};{lon2},{lat2}"
        f"?overview=full&geometries=geojson"
    )

    try:
        resp = requests.get(url)
        data = resp.json()
        rota_coords = data['routes'][0]['geometry']['coordinates']
        distancia_urbana = data['routes'][0]['distance']

        # Conversões para cálculo de trechos
        ponto_inicial_osrm = rota_coords[0]
        ponto_final_osrm = rota_coords[-1]

        # OSRM coords estão como [lon, lat] → inverter para [lat, lon]
        ponto_inicial = (ponto_inicial_osrm[1], ponto_inicial_osrm[0])
        ponto_final = (ponto_final_osrm[1], ponto_final_osrm[0])

        # Cálculos geodésicos adicionais
        distancia_inicial = geodesic(coord_origem, ponto_inicial).meters
        distancia_final = geodesic(coord_destino, ponto_final).meters

        # Distância total real
        distancia_total_real = distancia_inicial + distancia_urbana + distancia_final

        # Adicionar pontos reais à rota
        rota_coords.insert(0, [lon1, lat1])  # caixa real
        rota_coords.append([lon2, lat2])     # ponto real

        return rota_coords, distancia_total_real
    except Exception as e:
        print(f"Erro ao calcular rota OSRM: {e}")
        return [], 0

def gerar_kmz(nome_base, rota_coords, ponto_consultado, caixa_mais_proxima, caixa_mais_proxima_nome, nome_ponto_referencia=None):
    kml = simplekml.Kml()

    # Linha entre ponto e caixa
    linha = kml.newlinestring(name="Rota entre ponto e caixa")
    linha.coords = rota_coords
    linha.style.linestyle.color = simplekml.Color.red
    linha.style.linestyle.width = 4

    # Definir nome do ponto
    nome_ponto_kmz = nome_ponto_referencia if nome_ponto_referencia else "Ponto de Referência"

    # Ponto consultado
    ponto_ref = kml.newpoint(
        name=nome_ponto_kmz,
        coords=[ponto_consultado],
        description=f"Localização consultada: {ponto_consultado}",
    )
    ponto_ref.style.iconstyle.icon.href = "http://maps.google.com/mapfiles/kml/pushpin/ltblu-pushpin.png"

    # Caixa mais próxima
    caixa_ponto = kml.newpoint(
        name="{}".format(caixa_mais_proxima_nome),
        coords=[caixa_mais_proxima],
        description=f"Coordenadas: {caixa_mais_proxima}",
    )
    caixa_ponto.style.iconstyle.color = simplekml.Color.white
    caixa_ponto.style.iconstyle.scale = 1.2
    caixa_ponto.style.iconstyle.icon.href = "http://maps.google.com/mapfiles/kml/shapes/donut.png"

    # Salvar KMZ
    os.makedirs("saida_kmz", exist_ok=True)
    kml_path = os.path.join("saida_kmz", f"{nome_base}.kmz")
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
    rotas_para_kmz_unico = []

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
        rota_coords, distancia_real = calcular_rota_osrm(coord_caixa_proxima, coord_ponto)

        # Mover a atribuição de distancia_metros e tipo_cabo para cá
        if not rota_coords:
            distancia_real = menor_dist_geodesica

        distancia_metros = round(distancia_real, 2)
        tipo_cabo = "Drop" if distancia_metros < 250 else "Auto Sustentado"

        ponto_coords = (ponto['LONGITUDE'], ponto['LATITUDE'])
        caixa_coords = (caixa_proxima['Longitude'], caixa_proxima['Latitude'])
        nome_ponto = ponto.get('Nome', '').strip()
        nome_caixa = caixa_proxima["Sigla"]

        if rota_coords:
            rotas_para_kmz_unico.append({
                "rota_coords": rota_coords,
                "ponto_coords": ponto_coords,
                "nome_ponto": nome_ponto if nome_ponto else f"Ponto {index+1}",
                "caixa_coords": caixa_coords,
                "nome_caixa": nome_caixa,
                "viabilidade": 'Viável' if distancia_metros < limite_fibra else '',
                "tipo_cabo": tipo_cabo
            })
            kmz_path = "[Gerado em KMZ único]"
        else:
            kmz_path = ""

        resultados.append({
            'Nome do Ponto de Referência': nome_ponto,
            'Cidade do Ponto': ponto.get('Cidade', ''),
            'Estado do Ponto': ponto.get('Estado', ''),
            'Localização do Ponto': f"{ponto['LATITUDE']}, {ponto['LONGITUDE']}",
            'Localização da Caixa': f"{caixa_proxima['Latitude']}, {caixa_proxima['Longitude']}",
            'Identificador': nome_caixa,
            'Cidade': caixa_proxima['Cidade'],
            'Estado': caixa_proxima['Estado'],
            'Categoria': caixa_proxima['Pasta'],
            'Distância da Rota (m)': distancia_metros,
            'Tipo de Cabo': tipo_cabo,
            'Viabilidade': 'Viável' if distancia_metros < limite_fibra else '',
            'Download da Rota (KMZ)': kmz_path
        })

    if rotas_para_kmz_unico:
        nome_arquivo_unico = "rotas_completas"
        kmz_unico_path = gerar_kmz_unico(nome_arquivo_unico, rotas_para_kmz_unico)
        for resultado in resultados:
            resultado["Download da Rota (KMZ)"] = kmz_unico_path

    return pd.DataFrame(resultados)

def gerar_mapa_interativo(df_resultados, caminho_html):
    mapa = folium.Map(location=[-5.8, -36.6], zoom_start=8)
    marcadores = MarkerCluster().add_to(mapa)

    for _, linha in df_resultados.iterrows():
        lat_ponto, lon_ponto = map(float, linha['Localização do Ponto'].split(', '))
        lat_caixa, lon_caixa = map(float, linha['Localização da Caixa'].split(', '))

        rota_coords, _ = calcular_rota_osrm((lat_caixa, lon_caixa), (lat_ponto, lon_ponto))
        if rota_coords:
            rota_convertida = [(lat, lon) for lon, lat in rota_coords]
            folium.PolyLine(
                locations=rota_convertida,
                color='red', weight=3,
                tooltip=f"{linha['Distância da Rota (m)']} metros."
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


def gerar_kmz_unico(nome_base, rotas_info):
    kml = simplekml.Kml()

    for item in rotas_info:
        rota_coords = item["rota_coords"]
        ponto_consultado = item["ponto_coords"]
        nome_ponto = item["nome_ponto"]
        caixa_coords = item["caixa_coords"]
        nome_caixa = item["nome_caixa"]
        viabilidade = item.get("viabilidade", "")
        tipo_cabo = item.get("tipo_cabo", "")

        # Criar pasta para o ponto
        pasta = kml.newfolder(name=nome_ponto)

        # Cor da linha da rota
        if viabilidade != "Viável":
            cor_linha = simplekml.Color.red
        elif tipo_cabo == "Drop":
            cor_linha = simplekml.Color.rgb(0, 0, 255)  # azul marinho
        elif tipo_cabo == "Auto Sustentado":
            cor_linha = simplekml.Color.rgb(0, 255, 0)
        else:
            cor_linha = simplekml.Color.gray

        # Linha da rota dentro da pasta
        linha = pasta.newlinestring(name=f"Rota - {nome_ponto}")
        linha.coords = rota_coords
        linha.style.linestyle.color = cor_linha
        linha.style.linestyle.width = 4

        # Cor do marcador do ponto consultado
        cor_marcador = "ltblu-pushpin.png" if viabilidade == "Viável" else "ylw-pushpin.png"

        # Ponto consultado dentro da pasta
        ponto_ref = pasta.newpoint(
            name=nome_ponto,
            coords=[ponto_consultado],
            description=f"Localização consultada: {ponto_consultado}",
        )
        ponto_ref.style.iconstyle.icon.href = f"http://maps.google.com/mapfiles/kml/pushpin/{cor_marcador}"

        # Caixa dentro da pasta
        caixa_ponto = pasta.newpoint(
            name=nome_caixa,
            coords=[caixa_coords],
            description=f"Coordenadas: {caixa_coords}",
        )
        caixa_ponto.style.iconstyle.color = simplekml.Color.white
        caixa_ponto.style.iconstyle.scale = 1.2
        caixa_ponto.style.iconstyle.icon.href = "http://maps.google.com/mapfiles/kml/shapes/donut.png"

    os.makedirs("saida_kmz", exist_ok=True)
    kml_path = os.path.join("saida_kmz", f"{nome_base}.kmz")
    kml.savekmz(kml_path)
    return kml_path
