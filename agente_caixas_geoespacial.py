import pandas as pd
import requests
from geopy.distance import geodesic
import folium
from folium.plugins import MarkerCluster
import simplekml
import os

# === IMPORTS PARA A ESTRATÉGIA DO BUFFER GEOESPACIAL (POSTES) ===
from shapely.geometry import LineString
import geopandas as gpd
from pyproj import Transformer


def _utm_epsg_from_latlon(lat, lon):
    """
    Retorna o EPSG da zona UTM adequada para a coordenada (lat, lon).
    Usa WGS84 / UTM: 326xx (Hemisfério Norte) e 327xx (Hemisfério Sul).
    """
    zone = int((lon + 180) // 6) + 1
    if lat >= 0:
        return 32600 + zone
    else:
        return 32700 + zone


def _build_transformers(lat, lon):
    """
    Cria transformers entre WGS84 (EPSG:4326) <-> UTM local para a região de (lat, lon).
    Retorna (to_utm, to_wgs).
    """
    epsg = _utm_epsg_from_latlon(lat, lon)
    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    to_wgs = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    return to_utm, to_wgs


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


def gerar_rota_por_postes(rota_coords_osrm, df_postes, buffer_m=20, coord_caixa=None, coord_ponto=None):
    """
    Gera uma rota 'aderente a postes' usando buffer geoespacial sobre a rota do OSRM,
    conectando também Caixa -> 1º Poste e Último Poste -> Ponto de Referência.

    Parâmetros:
      - rota_coords_osrm: lista de [lon, lat] (como retorna o OSRM) incluindo origem/destino reais
      - df_postes: DataFrame com colunas Latitude/Longitude (case-insensitive) ou LATITUDE/LONGITUDE
      - buffer_m: largura do corredor em METROS ao redor da rota
      - coord_caixa: tupla (lat, lon) da caixa mais próxima
      - coord_ponto: tupla (lat, lon) do ponto de referência

    Retorna: (coords_rota_postes_wgs84, distancia_m)
      - coords_rota_postes_wgs84: lista ordenada de [lon, lat] começando na caixa (se informada),
        passando pelos postes e finalizando no ponto (se informado).
      - distancia_m: soma das distâncias euclidianas sequenciais (em metros) entre todos os segmentos.
    """
    if not rota_coords_osrm:
        return [], 0.0

    # Normaliza nomes de colunas de postes
    lower_map = {c.lower(): c for c in df_postes.columns}
    lat_col = lower_map.get('latitude') or lower_map.get('lat') or lower_map.get('y')
    lon_col = lower_map.get('longitude') or lower_map.get('lon') or lower_map.get('x')
    if lat_col is None or lon_col is None:
        if 'LATITUDE' in df_postes.columns and 'LONGITUDE' in df_postes.columns:
            lat_col, lon_col = 'LATITUDE', 'LONGITUDE'
        else:
            # Sem colunas reconhecíveis
            return [], 0.0

    # Ponto médio para escolher UTM local
    mid_idx = len(rota_coords_osrm) // 2
    ref_lon, ref_lat = rota_coords_osrm[mid_idx]
    to_utm, to_wgs = _build_transformers(ref_lat, ref_lon)
    epsg = _utm_epsg_from_latlon(ref_lat, ref_lon)

    # Projeta rota OSRM para UTM e cria buffer em metros
    line_utm = LineString([to_utm.transform(lon, lat) for lon, lat in rota_coords_osrm])
    buffer_utm = line_utm.buffer(buffer_m)

    # GDF de postes em WGS84 -> reprojeta para UTM
    gdf_postes = gpd.GeoDataFrame(
        df_postes.copy(),
        geometry=gpd.points_from_xy(df_postes[lon_col], df_postes[lat_col]),
        crs="EPSG:4326",
    ).to_crs(f"EPSG:{epsg}")

    # Seleciona postes dentro do buffer
    postes_no_corredor = gdf_postes[gdf_postes.geometry.within(buffer_utm)]
    if postes_no_corredor.empty:
        # Mesmo sem postes no buffer, conectamos caixa->ponto se ambos forem informados
        coords_full_wgs = []
        if coord_caixa is not None:
            coords_full_wgs.append((to_utm.transform(coord_caixa[1], coord_caixa[0])))  # (x,y) UTM
        if coord_ponto is not None:
            coords_full_wgs.append((to_utm.transform(coord_ponto[1], coord_ponto[0])))
        # Distância direta (se ambos fornecidos)
        distancia_m = 0.0
        if len(coords_full_wgs) >= 2:
            x1, y1 = coords_full_wgs[0]
            x2, y2 = coords_full_wgs[-1]
            distancia_m = ((x2-x1)**2 + (y2-y1)**2) ** 0.5
        # Retorna em WGS84 (lon,lat)
        coords_full_lonlat = []
        if coord_caixa is not None:
            lon, lat = coord_caixa[1], coord_caixa[0]
            coords_full_lonlat.append((lon, lat))
        if coord_ponto is not None:
            lon, lat = coord_ponto[1], coord_ponto[0]
            coords_full_lonlat.append((lon, lat))
        return coords_full_lonlat, distancia_m

    # Ordena pela progressão ao longo da linha (em UTM)
    postes_no_corredor["dist_along"] = postes_no_corredor.geometry.apply(lambda p: line_utm.project(p))
    postes_ord = postes_no_corredor.sort_values("dist_along")

    # Constrói a sequência final em UTM: Caixa -> Postes -> Ponto
    coords_full_utm = []

    # Caixa
    if coord_caixa is not None:
        coords_full_utm.append(to_utm.transform(coord_caixa[1], coord_caixa[0]))  # (x,y)

    # Postes
    for geom in postes_ord.geometry:
        coords_full_utm.append((geom.x, geom.y))

    # Ponto final
    if coord_ponto is not None:
        coords_full_utm.append(to_utm.transform(coord_ponto[1], coord_ponto[0]))

    # Distância total
    distancia_m = 0.0
    for i in range(1, len(coords_full_utm)):
        x1, y1 = coords_full_utm[i-1]
        x2, y2 = coords_full_utm[i]
        distancia_m += ((x2-x1)**2 + (y2-y1)**2) ** 0.5

    # Reprojetar para WGS84 (lon,lat) para consumo do restante do app/KMZ
    coords_full_lonlat = [to_wgs.transform(x, y) for (x, y) in coords_full_utm]

    return coords_full_lonlat, distancia_m


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


def analisar_distancia_entre_pontos(df_pontos, df_caixas, limite_fibra=350, df_postes=None, buffer_postes_m=None, usar_postes=False):
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

        # Fallback para distância se OSRM falhar
        if not rota_coords:
            distancia_real = menor_dist_geodesica

        # Distância oficial (padrão = OSRM com ajustes)
        distancia_metros = round(distancia_real, 2)
        tipo_cabo = "Drop" if distancia_metros < 250 else "Auto Sustentado"

        ponto_coords = (ponto['LONGITUDE'], ponto['LATITUDE'])
        caixa_coords = (caixa_proxima['Longitude'], caixa_proxima['Latitude'])
        nome_ponto = ponto.get('Nome', '').strip() if isinstance(ponto.get('Nome', ''), str) else ''
        nome_caixa = caixa_proxima["Sigla"]

        # --- ROTA POR POSTES (opcional) ---
        rota_postes_coords = []
        dist_postes_m = None
        if usar_postes and rota_coords and df_postes is not None and buffer_postes_m:
            try:
                rota_postes_coords, dist_postes_m = gerar_rota_por_postes(
                    rota_coords, df_postes, buffer_m=float(buffer_postes_m),
                    coord_caixa=coord_caixa_proxima, coord_ponto=coord_ponto
                )
                # Se quiser que a distância oficial passe a ser a via postes, descomente:
                # if dist_postes_m and dist_postes_m > 0:
                #     distancia_metros = round(dist_postes_m, 2)
                #     tipo_cabo = "Drop" if distancia_metros < 250 else "Auto Sustentado"
            except Exception as e:
                print(f"[Postes] Falha ao gerar rota por postes: {e}")

        if rota_coords:
            rotas_para_kmz_unico.append({
                "rota_coords": rota_coords,
                "ponto_coords": ponto_coords,
                "nome_ponto": nome_ponto if nome_ponto else f"Ponto {index+1}",
                "caixa_coords": caixa_coords,
                "nome_caixa": nome_caixa,
                "viabilidade": 'Viável' if distancia_metros < limite_fibra else '',
                "tipo_cabo": tipo_cabo,
                # Guarda info da rota por postes (para uso no KMZ)
                "rota_postes_coords": rota_postes_coords,
                "dist_postes_m": dist_postes_m
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
            'Distância via Postes (m)': round(dist_postes_m, 2) if dist_postes_m else None,
            'Rota via Postes?': 'Sim' if rota_postes_coords else 'Não',
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

        # Rota OSRM
        rota_coords, _ = calcular_rota_osrm((lat_caixa, lon_caixa), (lat_ponto, lon_ponto))
        if rota_coords:
            rota_convertida = [(lat, lon) for lon, lat in rota_coords]
            folium.PolyLine(
                locations=rota_convertida,
                color='red', weight=4,
                tooltip=f"Rota OSRM: {linha['Distância da Rota (m)']} metros."
            ).add_to(mapa)

            # Rota via Postes
            if linha.get('Rota via Postes?') == 'Sim' and linha.get('Distância via Postes (m)'):
                rota_postes_coords = linha.get('rota_postes_coords', [])
                if rota_postes_coords:
                    rota_postes_convertida = [(lat, lon) for lon, lat in rota_postes_coords]
                    folium.PolyLine(
                        locations=rota_postes_convertida,
                        color='blue', weight=4,
                        tooltip=f"Rota Via Postes: {linha['Distância via Postes (m)']} metros"
                    ).add_to(mapa)
        else:
            folium.PolyLine(
                locations=[[lat_ponto, lon_ponto], [lat_caixa, lon_caixa]],
                color='gray', weight=2, dash_array="5,5",
                tooltip="Rota linear (falha OSRM)"
            ).add_to(mapa)

        # Marcadores
        folium.Marker(
            location=[lat_caixa, lon_caixa],
            tooltip=f"{linha['Identificador']}",
            icon=folium.Icon(color='green', icon='hdd', prefix='fa')
        ).add_to(marcadores)

        folium.Marker(
            location=[lat_ponto, lon_ponto],
            tooltip=f"{linha['Nome do Ponto de Referência']}",
            icon=folium.Icon(color='blue', icon='info-sign')
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
        rota_postes_coords = item.get("rota_postes_coords") or []
        dist_postes_m = item.get("dist_postes_m")  # pode ser None

        # Criar pasta para o ponto
        pasta = kml.newfolder(name=nome_ponto)

        # --- Linha da rota principal (OSRM) ---
        # Cor por status (mantém sua lógica existente)
        if viabilidade != "Viável":
            cor_linha = simplekml.Color.red
        elif tipo_cabo == "Drop":
            cor_linha = simplekml.Color.rgb(0, 0, 255)  # azul marinho
        elif tipo_cabo == "Auto Sustentado":
            cor_linha = simplekml.Color.rgb(0, 255, 0)
        else:
            cor_linha = simplekml.Color.gray

        linha = pasta.newlinestring(name=f"Rota OSRM - {nome_ponto}")
        linha.coords = rota_coords
        linha.style.linestyle.color = cor_linha
        linha.style.linestyle.width = 4

        # --- Linha da rota via Postes (se existir) ---
        if rota_postes_coords:
            linha_postes = pasta.newlinestring(name=f"Rota por Postes - {nome_ponto}")
            linha_postes.coords = rota_postes_coords
            # cor distinta (ciano com leve transparência)
            # Marcadores de postes
            for idx, (lon, lat) in enumerate(rota_postes_coords):
                # pular o primeiro e o último (caixa e ponto de referência)
                if idx == 0 or idx == len(rota_postes_coords)-1:
                    continue
                p_poste = pasta.newpoint(coords=[(lon, lat)])
                p_poste.style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png'
                p_poste.style.iconstyle.scale = 0.8
            linha_postes.style.linestyle.color = simplekml.Color.changealphaint(180, simplekml.Color.cyan)
            linha_postes.style.linestyle.width = 4
            if dist_postes_m:
                linha_postes.description = f"Rota por postes (~{round(dist_postes_m, 2)} m)"
            else:
                linha_postes.description = "Rota por postes"

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