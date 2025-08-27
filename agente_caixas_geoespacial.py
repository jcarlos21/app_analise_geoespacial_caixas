import pandas as pd
import requests
from geopy.distance import geodesic
import folium
from folium.plugins import MarkerCluster
import simplekml
import os

# === IMPORTS PARA A ESTRATÉGIA DO BUFFER GEOESPACIAL (POSTES) ===
from shapely.geometry import LineString, Point
import geopandas as gpd
from pyproj import Transformer

# === NOVOS IMPORTS (necessários para as 3 camadas) ===
import math
import numpy as np

def _utm_epsg_from_latlon(lat, lon):
    zone = int((lon + 180) // 6) + 1
    if lat >= 0:
        return 32600 + zone
    else:
        return 32700 + zone

def _build_transformers(lat, lon):
    epsg = _utm_epsg_from_latlon(lat, lon)
    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    to_wgs = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    return to_utm, to_wgs

def calcular_rota_osrm(coord_origem, coord_destino):
    lat1, lon1 = coord_origem  # caixa
    lat2, lon2 = coord_destino  # ponto consultado
    url = (
        f"https://router.project-osrm.org/route/v1/foot/"
        f"{lon1},{lat1};{lon2},{lat2}"
        f"?overview=full&geometries=geojson&alternatives=true&steps=false"
    )
    try:
        resp = requests.get(url)
        data = resp.json()

        # === MELHORIA #1: escolher a melhor entre alternativas ===
        rotas = data.get('routes', [])
        if not rotas:
            raise ValueError("Nenhuma rota retornada pelo OSRM.")

        melhor = min(rotas, key=lambda r: r.get('distance', float('inf')))
        rota_coords = melhor['geometry']['coordinates']
        distancia_urbana = melhor['distance']

        ponto_inicial_osrm = rota_coords[0]
        ponto_final_osrm = rota_coords[-1]

        ponto_inicial = (ponto_inicial_osrm[1], ponto_inicial_osrm[0])
        ponto_final = (ponto_final_osrm[1], ponto_final_osrm[0])

        distancia_inicial = geodesic(coord_origem, ponto_inicial).meters
        distancia_final = geodesic(coord_destino, ponto_final).meters

        distancia_total_real = distancia_inicial + distancia_urbana + distancia_final

        # Garante pontas reais (caixa e ponto)
        rota_coords.insert(0, [lon1, lat1])  # caixa real
        rota_coords.append([lon2, lat2])     # ponto real

        return rota_coords, distancia_total_real
    except Exception as e:
        print(f"[OSRM] Erro ao calcular rota: {e}. Verifique conectividade e formato das coordenadas.")
        return [], 0

# === NOVO: OSRM bidirecional (A→B e B→A; escolhe a menor) ===
def calcular_rota_osrm_bidir(coord_a, coord_b):
    """
    Calcula rota A→B e B→A usando calcular_rota_osrm (com alternatives=true)
    e retorna a menor. Se a menor vier como B→A, a geometria é revertida
    para manter a orientação A→B no restante do pipeline.
    """
    rota_ab, dist_ab = calcular_rota_osrm(coord_a, coord_b)
    rota_ba, dist_ba = calcular_rota_osrm(coord_b, coord_a)

    if not rota_ab and not rota_ba:
        return [], 0

    if rota_ab and (not rota_ba or dist_ab <= dist_ba):
        return rota_ab, dist_ab

    # Melhor foi B→A: reorienta para A→B
    rota_ba_reorientada = list(reversed(rota_ba))
    return rota_ba_reorientada, dist_ba

# ---------- Funções auxiliares p/ 3 camadas ----------
def _unit(vx, vy):
    n = math.hypot(vx, vy)
    if n == 0:
        return 0.0, 0.0
    return vx / n, vy / n

def _signed_offset(line_utm: LineString, p: Point):
    s = line_utm.project(p)
    on = line_utm.interpolate(s)
    s2 = min(s + 0.5, line_utm.length)
    on2 = line_utm.interpolate(s2)
    tx, ty = _unit(on2.x - on.x, on2.y - on.y)
    nx, ny = -ty, tx
    d = p.distance(on)
    vx, vy = (p.x - on.x), (p.y - on.y)
    d_signed = vx * nx + vy * ny
    return s, d, d_signed

def _angle(p0, p1, p2):
    v1 = (p0[0]-p1[0], p0[1]-p1[1])
    v2 = (p2[0]-p1[0], p2[1]-p1[1])
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cosang = (v1[0]*v2[0] + v1[1]*v2[1]) / (n1*n2)
    cosang = max(-1.0, min(1.0, cosang))
    ang = math.degrees(math.acos(cosang))
    return ang

def gerar_rota_por_postes(
    rota_coords_osrm,
    df_postes,
    buffer_m=20,
    coord_caixa=None,
    coord_ponto=None,
    max_lateral=15.0,
    frac_inicial=0.2,
    angulo_max=60.0,
    delta_s_min=10.0,
    salto_lateral_max=0.0,
    espacamento_min=0,
    dp_tol=0,
    gap_s=70.0,
    # === MELHORIA #2: parâmetros do corredor híbrido ===
    modo_corredor="uniao",     # "osrm", "reta" ou "uniao"
    buffer_reta_m=None         # se None, usa buffer_m
):
    if not rota_coords_osrm:
        return [], 0.0

    # Normaliza nomes de colunas de postes
    lower_map = {c.lower(): c for c in df_postes.columns}
    lat_col = lower_map.get('latitude') or lower_map.get('lat') or lower_map.get('y')
    lon_col = lower_map.get('longitude') or lower_map.get('lon') or lower_map.get('x')
    if lat_col is None or lon_col is None:
        print("[POSTES] Colunas de latitude/longitude não foram encontradas no arquivo de postes.")
        return [], 0.0

    # Ponto médio para escolher UTM local
    mid_idx = len(rota_coords_osrm) // 2
    ref_lon, ref_lat = rota_coords_osrm[mid_idx]
    to_utm, to_wgs = _build_transformers(ref_lat, ref_lon)
    epsg = _utm_epsg_from_latlon(ref_lat, ref_lon)

    # Projeta rota OSRM para UTM e cria buffer
    line_osrm_utm = LineString([to_utm.transform(lon, lat) for lon, lat in rota_coords_osrm])
    buf_osrm = line_osrm_utm.buffer(buffer_m)

    # === MELHORIA #2: linha reta e união de buffers ===
    if coord_caixa is not None and coord_ponto is not None:
        pA = to_utm.transform(coord_caixa[1], coord_caixa[0])
        pB = to_utm.transform(coord_ponto[1], coord_ponto[0])
        line_reta_utm = LineString([pA, pB])
        buf_reta = line_reta_utm.buffer(buffer_reta_m or buffer_m)
    else:
        line_reta_utm = None
        buf_reta = None

    if modo_corredor == "osrm":
        corredor_utm = buf_osrm
    elif modo_corredor == "reta" and buf_reta is not None:
        corredor_utm = buf_reta
    else:  # "uniao" (padrão)
        corredor_utm = buf_osrm.union(buf_reta) if buf_reta is not None else buf_osrm

    # GDF de postes em WGS84 -> reprojeta p/ UTM
    gdf_postes = gpd.GeoDataFrame(
        df_postes.copy(),
        geometry=gpd.points_from_xy(df_postes[lon_col], df_postes[lat_col]),
        crs="EPSG:4326",
    ).to_crs(f"EPSG:{epsg}")

    # Seleciona postes dentro do corredor (antes era apenas buffer da OSRM)
    postes_no_corredor = gdf_postes[gdf_postes.geometry.within(corredor_utm)]
    if postes_no_corredor.empty:
        coords_full_utm = []
        if coord_caixa is not None:
            coords_full_utm.append(to_utm.transform(coord_caixa[1], coord_caixa[0]))
        if coord_ponto is not None:
            coords_full_utm.append(to_utm.transform(coord_ponto[1], coord_ponto[0]))
        distancia_m = 0.0
        for i in range(1, len(coords_full_utm)):
            x1, y1 = coords_full_utm[i-1]
            x2, y2 = coords_full_utm[i]
            distancia_m += math.hypot(x2-x1, y2-y1)
        coords_full_lonlat = [to_wgs.transform(x, y) for (x, y) in coords_full_utm]
        return coords_full_lonlat, distancia_m

    # ---------- Cálculos s, d, d_signed ----------
    rows = []
    for _, r in postes_no_corredor.iterrows():
        p = r.geometry
        s, d, d_signed = _signed_offset(line_osrm_utm, p)
        rows.append((s, d, d_signed, p.x, p.y))
    arr = np.array(rows)  # colunas: s, d, d_signed, x, y
    arr = arr[arr[:, 0].argsort()]

    # ---------- 1) Lado dominante ----------
    n = len(arr)
    n_ini = max(1, int(frac_inicial * n))
    d_signed_ini = arr[:n_ini, 2]
    med = np.median(d_signed_ini)
    if abs(med) < 1e-6:
        pos = np.sum(arr[:, 2] > 0)
        neg = np.sum(arr[:, 2] < 0)
        med = 1.0 if pos >= neg else -1.0
    sign_dom = 1.0 if med >= 0 else -1.0

    mask_lado = np.sign(arr[:, 2]) == sign_dom
    mask_lateral = arr[:, 1] <= max_lateral
    base_arr = arr[mask_lado & mask_lateral]

    # ---------- 3a) Tolerância a gaps ----------
    if len(base_arr) >= 2:
        s_base = base_arr[:, 0]
        gaps_idx = np.where(np.diff(s_base) > gap_s)[0]
        for gi in gaps_idx:
            s_a, s_b = s_base[gi], s_base[gi+1]
            relax = max_lateral * 1.2
            win_mask = (arr[:, 0] > s_a) & (arr[:, 0] < s_b) & (arr[:, 1] <= relax)
            candidatos = arr[win_mask]
            if len(candidatos) > 0:
                base_arr = np.vstack([base_arr, candidatos])
        base_arr = base_arr[base_arr[:, 0].argsort()]

    # ---------- 2) Regras anti-serrilha ----------
    seq = base_arr.tolist()

    def _filtra_angulos(seq_pts, ang_max=60.0, ds_min=10.0):
        if len(seq_pts) < 3:
            return seq_pts
        res = [seq_pts[0]]
        for i in range(1, len(seq_pts)-1):
            prev = res[-1]
            cur = seq_pts[i]
            nxt = seq_pts[i+1]
            p0 = (prev[3], prev[4])
            p1 = (cur[3], cur[4])
            p2 = (nxt[3], nxt[4])
            ang = _angle(p0, p1, p2)
            ds = nxt[0] - prev[0]
            if ang > ang_max and ds < ds_min:
                continue
            res.append(cur)
        res.append(seq_pts[-1])
        return res

    def _filtra_salto_lateral(seq_pts, salto_max=0.0, ds_min=10.0):
        if len(seq_pts) < 2:
            return seq_pts
        res = [seq_pts[0]]
        for i in range(1, len(seq_pts)):
            prev = res[-1]
            cur = seq_pts[i]
            salto = abs(cur[2] - prev[2])
            ds = cur[0] - prev[0]
            if salto > salto_max and ds < ds_min:
                continue
            res.append(cur)
        return res

    def _thinning(seq_pts, min_s=0):
        if len(seq_pts) < 2:
            return seq_pts
        res = [seq_pts[0]]
        last_s = seq_pts[0][0]
        for i in range(1, len(seq_pts)-1):
            if seq_pts[i][0] - last_s >= min_s:
                res.append(seq_pts[i])
                last_s = seq_pts[i][0]
        res.append(seq_pts[-1])
        return res

    seq = _filtra_angulos(seq, ang_max=angulo_max, ds_min=delta_s_min)
    seq = _filtra_salto_lateral(seq, salto_max=salto_lateral_max, ds_min=delta_s_min)
    seq = _thinning(seq, min_s=espacamento_min)

    coords_full_utm = []
    if coord_caixa is not None:
        coords_full_utm.append(to_utm.transform(coord_caixa[1], coord_caixa[0]))
    for s, d, dsig, x, y in seq:
        coords_full_utm.append((x, y))
    if coord_ponto is not None:
        coords_full_utm.append(to_utm.transform(coord_ponto[1], coord_ponto[0]))

    if len(coords_full_utm) >= 3:
        line = LineString(coords_full_utm)
        line_simpl = line.simplify(dp_tol, preserve_topology=False)
        coords_full_utm = list(line_simpl.coords)

    distancia_m = 0.0
    for i in range(1, len(coords_full_utm)):
        x1, y1 = coords_full_utm[i-1]
        x2, y2 = coords_full_utm[i]
        distancia_m += math.hypot(x2-x1, y2-y1)

    coords_full_lonlat = [to_wgs.transform(x, y) for (x, y) in coords_full_utm]
    return coords_full_lonlat, distancia_m

def gerar_kmz(nome_base, rota_coords, ponto_consultado, caixa_mais_proxima, caixa_mais_proxima_nome, nome_ponto_referencia=None):
    kml = simplekml.Kml()
    linha = kml.newlinestring(name="Rota entre ponto e caixa")
    linha.coords = rota_coords
    linha.style.linestyle.color = simplekml.Color.red
    linha.style.linestyle.width = 4

    nome_ponto_kmz = nome_ponto_referencia if nome_ponto_referencia else "Ponto de Referência"

    ponto_ref = kml.newpoint(
        name=nome_ponto_kmz,
        coords=[ponto_consultado],
        description=f"Localização consultada: {ponto_consultado}",
    )
    ponto_ref.style.iconstyle.icon.href = "http://maps.google.com/mapfiles/kml/pushpin/ltblu-pushpin.png"

    caixa_ponto = kml.newpoint(
        name=f"{caixa_mais_proxima_nome}",
        coords=[caixa_mais_proxima],
        description=f"Coordenadas: {caixa_mais_proxima}",
    )
    caixa_ponto.style.iconstyle.color = simplekml.Color.white
    caixa_ponto.style.iconstyle.scale = 1.2
    caixa_ponto.style.iconstyle.icon.href = "http://maps.google.com/mapfiles/kml/shapes/donut.png"

    os.makedirs("saida_kmz", exist_ok=True)
    kml_path = os.path.join("saida_kmz", f"{nome_base}.kmz")
    kml.savekmz(kml_path)
    return kml_path

# --------- NOVO: normalização robusta do df_caixas ---------
def _normalize_caixas_df(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower().strip(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None
    mapping = {}
    lat = pick('latitude','lat','y')
    lon = pick('longitude','lon','x')
    cidade = pick('cidade','município','municipio')
    estado = pick('estado','uf','sigla uf')
    pasta = pick('pasta','categoria')
    sigla = pick('sigla','identificador','id','nome','caixa','id_caixa')
    if lat: mapping[lat] = 'Latitude'
    if lon: mapping[lon] = 'Longitude'
    if cidade: mapping[cidade] = 'Cidade'
    if estado: mapping[estado] = 'Estado'
    if pasta: mapping[pasta] = 'Pasta'
    if sigla: mapping[sigla] = 'Sigla'
    df2 = df.rename(columns=mapping).copy()

    # validações mínimas
    for req in ['Latitude', 'Longitude']:
        if req not in df2.columns:
            raise KeyError(f"Coluna obrigatória ausente no arquivo de caixas: '{req}'.")
    return df2

def analisar_distancia_entre_pontos(
    df_pontos, df_caixas, limite_fibra=350, df_postes=None, buffer_postes_m=None,
    usar_postes=False, k_top=3  # <<=== mantém Top-K configurável
):
    # normaliza df_pontos (como já fazia)
    df_pontos.columns = df_pontos.columns.str.strip().str.upper()
    df_pontos.rename(columns={
        'NOME': 'Nome',
        'NOME MUNICÍPIO': 'Cidade',
        'SIGLA UF': 'Estado',
        'LATITUDE': 'LATITUDE',
        'LONGITUDE': 'LONGITUDE'
    }, inplace=True)

    # ✅ normaliza df_caixas (novo)
    df_caixas = _normalize_caixas_df(df_caixas)

    resultados = []
    rotas_para_kmz_unico = []

    if df_caixas.empty:
        return pd.DataFrame([])

    K_TOP = max(1, int(k_top))  # garante valor mínimo 1

    for index, ponto in df_pontos.iterrows():
        coord_ponto = (ponto['LATITUDE'], ponto['LONGITUDE'])

        # --- Pré-seleção por geodésica: pega as K caixas mais próximas
        def _dist_geo_row(row):
            return geodesic(coord_ponto, (row['Latitude'], row['Longitude'])).meters

        df_tmp = df_caixas.copy()
        df_tmp['__dist_geo__'] = df_tmp.apply(_dist_geo_row, axis=1)
        candidatas = df_tmp.nsmallest(K_TOP, '__dist_geo__')

        # --- Entre as K candidatas, escolhe a menor rota real (OSRM) usando BIDIR
        melhor = None
        for _, cand in candidatas.iterrows():
            coord_caixa_cand = (cand['Latitude'], cand['Longitude'])

            # >>> alteração: usa cálculo bidirecional
            rota_coords_cand, distancia_real_cand = calcular_rota_osrm_bidir(coord_caixa_cand, coord_ponto)

            if not rota_coords_cand:
                # fallback: usa geodésica dessa candidata
                distancia_real_cand = cand['__dist_geo__']

            if (melhor is None) or (distancia_real_cand < melhor['dist']):
                melhor = {
                    'caixa': cand,
                    'rota': rota_coords_cand,
                    'dist': distancia_real_cand
                }

        if melhor is None:
            continue

        caixa_proxima = melhor['caixa']
        coord_caixa_proxima = (caixa_proxima['Latitude'], caixa_proxima['Longitude'])
        rota_coords = melhor['rota']
        distancia_real = melhor['dist']

        # Se por algum motivo a rota não veio, usa geodésica
        if not rota_coords:
            distancia_real = geodesic(coord_ponto, coord_caixa_proxima).meters

        distancia_metros = round(distancia_real, 2)
        tipo_cabo = "Drop" if distancia_metros < 250 else "Auto Sustentado"

        ponto_coords = (ponto['LONGITUDE'], ponto['LATITUDE'])
        caixa_coords = (caixa_proxima['Longitude'], caixa_proxima['Latitude'])
        nome_ponto = ponto.get('Nome', '').strip() if isinstance(ponto.get('Nome', ''), str) else ''
        nome_caixa = caixa_proxima.get("Sigla", "")

        rota_postes_coords = []
        dist_postes_m = None
        if usar_postes and rota_coords and df_postes is not None and buffer_postes_m:
            try:
                # MELHORIA #2 já aplicada dentro da função (corredor híbrido)
                rota_postes_coords, dist_postes_m = gerar_rota_por_postes(
                    rota_coords, df_postes, buffer_m=float(buffer_postes_m),
                    coord_caixa=coord_caixa_proxima, coord_ponto=coord_ponto
                )
            except Exception as e:
                print(f"[POSTES] Falha ao gerar rota por postes: {e}")

        if rota_coords:
            rotas_para_kmz_unico.append({
                "rota_coords": rota_coords,
                "ponto_coords": ponto_coords,
                "nome_ponto": nome_ponto if nome_ponto else f"Ponto {index+1}",
                "caixa_coords": caixa_coords,
                "nome_caixa": nome_caixa,
                "viabilidade": 'Viável' if distancia_metros < limite_fibra else '',
                "tipo_cabo": tipo_cabo,
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
            'Cidade': caixa_proxima.get('Cidade', ''),
            'Estado': caixa_proxima.get('Estado', ''),
            'Categoria': caixa_proxima.get('Pasta', ''),
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

        # >>> alteração: usa cálculo bidirecional também no mapa
        rota_coords, _ = calcular_rota_osrm_bidir((lat_caixa, lon_caixa), (lat_ponto, lon_ponto))
        if rota_coords:
            rota_convertida = [(lat, lon) for lon, lat in rota_coords]
            folium.PolyLine(
                locations=rota_convertida,
                color='red', weight=4,
                tooltip=f"Rota OSRM: {linha['Distância da Rota (m)']} metros."
            ).add_to(mapa)

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
        dist_postes_m = item.get("dist_postes_m")

        pasta = kml.newfolder(name=nome_ponto)

        if viabilidade != "Viável":
            cor_linha = simplekml.Color.red
        elif tipo_cabo == "Drop":
            cor_linha = simplekml.Color.rgb(0, 0, 255)
        elif tipo_cabo == "Auto Sustentado":
            cor_linha = simplekml.Color.rgb(0, 255, 0)
        else:
            cor_linha = simplekml.Color.gray

        linha = pasta.newlinestring(name=f"Rota OSRM - {nome_ponto}")
        linha.coords = rota_coords
        linha.style.linestyle.color = cor_linha
        linha.style.linestyle.width = 4

        if rota_postes_coords:
            linha_postes = pasta.newlinestring(name=f"Rota por Postes - {nome_ponto}")
            linha_postes.coords = rota_postes_coords
            for idx, (lon, lat) in enumerate(rota_postes_coords):
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

        cor_marcador = "ltblu-pushpin.png" if viabilidade == "Viável" else "ylw-pushpin.png"

        ponto_ref = pasta.newpoint(
            name=nome_ponto,
            coords=[ponto_consultado],
            description=f"Localização consultada: {ponto_consultado}",
        )
        ponto_ref.style.iconstyle.icon.href = f"http://maps.google.com/mapfiles/kml/pushpin/{cor_marcador}"

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
