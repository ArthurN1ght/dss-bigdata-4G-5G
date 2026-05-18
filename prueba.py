import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

# --- MULTIPLATAFORMA: PIPELINE ML SEGURO PARA ENTREGAS DE CURSOS ---
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(page_title="NMS DSS Expert v10 - Pipeline ML", layout="wide")
st.title("📡 Analizador NMS Big Data: DSS 4G/5G con Pipeline de Machine Learning")

# --- 1. CONFIGURACIÓN SIDEBAR ---
with st.sidebar:
    st.header("📂 Carga de Archivos")
    file_trafico = st.file_uploader("Cargar Data11.csv", type=["csv"])
    file_geo = st.file_uploader("Cargar 2025 Sites 1900.kml", type=["kml"])
    st.divider()
    st.header("⚙️ Parámetros del Algoritmo")
    gbr_min = st.slider("Piso Mínimo Garantizado (Mbps):", 15, 40, 15)
    ratio_5g = st.slider("% Recursos base para 5G (DSS):", 10, 90, 60)
    st.caption("El modelo de ML reescribirá este ratio dinámicamente en las zonas donde detecte la necesidad de DSS Automático.")

# --- 2. EXTRACCIÓN GEOGRÁFICA (KML) ---
def extraer_geo_kml(file):
    try:
        content = file.read()
        root = ET.fromstring(content)
        ns = {'kml': 'http://www.opengis.net/kml/2.2'}
        geo_map = {}
        for pm in root.findall('.//kml:Placemark', ns):
            name = pm.find('kml:name', ns)
            coord = pm.find('.//kml:coordinates', ns)
            if name is not None and coord is not None:
                c_split = coord.text.strip().split(',')
                geo_map[name.text.strip().upper()] = {
                    "lon": float(c_split[0]),
                    "lat": float(c_split[1])
                }
        return geo_map
    except Exception as e:
        st.error(f"Error en KML: {e}")
        return None

# --- 3. PIPELINE DE PROCESAMIENTO Y MACHINE LEARNING ---
if file_trafico and file_geo:
    geo_map = extraer_geo_kml(file_geo)
    
    # Lectura y limpieza del dataset de tráfico
    pd_raw = pd.read_csv(file_trafico, sep=';', on_bad_lines='skip')
    pd_raw.columns = [str(c).strip() for c in pd_raw.columns]
    
    col_site = 'ManagedElement Name'
    col_traf = 'LTE PS Traffic'
    
    # --- ETAPA 1 DEL PIPELINE: INGESTIÓN Y AGREGACIÓN EN LOTE ---
    site_metrics = pd_raw.dropna(subset=[col_site, col_traf]).groupby(col_site).agg(
        Trafico_Promedio=(col_traf, 'mean'),
        Variabilidad_Trafico=(col_traf, 'std'),
        Trafico_Pico=(col_traf, 'max')
    ).fillna(0).reset_index()
    
    # --- ETAPA 2 DEL PIPELINE: INGENIERÍA DE CARACTERÍSTICAS (FEATURE ENGINEERING) ---
    features = ["Trafico_Promedio", "Variabilidad_Trafico", "Trafico_Pico"]
    X = site_metrics[features].values
    
    # Normalización de los datos para el algoritmo (Equivalente al StandardScaler de Spark)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # --- ETAPA 3 DEL PIPELINE: MODELAMIENTO PREDICTIVO (K-MEANS) ---
    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
    site_metrics["Cluster_DSS"] = kmeans.fit_predict(X_scaled)
    
    # Mapeo analítico basado en la carga del cluster
    order_centers = site_metrics.groupby("Cluster_DSS")["Trafico_Promedio"].mean().sort_values().index
    cluster_mapping = {
        order_centers[0]: "Dominancia 4G (Bajo Tráfico)", 
        order_centers[1]: "DSS Automático Requerido (Tráfico Fluctuante)", 
        order_centers[2]: "Dominancia 5G (Alta Carga/Nodos Core)"
    }
    site_metrics["Estado_DSS"] = site_metrics["Cluster_DSS"].map(cluster_mapping)

    # --- ETAPA 4 DEL PIPELINE: VÍNCULO GEOGRÁFICO ---
    mapping = {}
    sitios_validos = []
    
    for _, row in site_metrics.iterrows():
        s_csv = row[col_site]
        s_base = str(s_csv).upper().split('_')[0].split(' ')[0]
        
        for s_kml in geo_map.keys():
            if s_base in s_kml or s_kml in s_base:
                mapping[s_csv] = geo_map[s_kml]
                sitios_validos.append(s_csv)
                break
                
    if sitios_validos:
        st.success(f"✅ Pipeline ejecutado con éxito. Clasificados {len(set(sitios_validos))} nodos de red.")
        
        site_metrics = site_metrics[site_metrics[col_site].isin(sitios_validos)].copy()
        
        # --- ETAPA 5 DEL PIPELINE: MOTOR DINÁMICO DE SIMULACIÓN DE TRÁFICO ---
        records = []
        ahora = datetime.now()
        max_w = site_metrics["Trafico_Promedio"].max() if site_metrics["Trafico_Promedio"].max() > 0 else 1
        
        for h in range(24 * 7):  # Ventana analítica de 7 días
            dt = ahora - timedelta(hours=h)
            hora_factor = 0.5 * np.sin((dt.hour - 10) * np.pi / 12) + 0.6
            
            for _, row in site_metrics.iterrows():
                s = row[col_site]
                pos = mapping[s]
                estado = row["Estado_DSS"]
                peso_rel = max(row["Trafico_Promedio"] / max_w, 0.1)
                
                if estado == "Dominancia 4G (Bajo Tráfico)":
                    dss_4g_factor, dss_5g_factor = 1.4, 0.6
                    color = [52, 152, 219, 180]  # Azul
                elif estado == "Dominancia 5G (Alta Carga/Nodos Core)":
                    dss_4g_factor, dss_5g_factor = 0.4, 1.6
                    color = [46, 204, 113, 180]  # Verde
                else: 
                    dss_4g_factor = 1.0 + 0.4 * np.cos(dt.hour * np.pi / 6)
                    dss_5g_factor = 1.0 - 0.4 * np.cos(dt.hour * np.pi / 6)
                    color = [241, 196, 15, 200]  # Amarillo (Asignación dinámica automática)
                
                for tech in ['4G', '5G']:
                    fluctuacion = np.random.uniform(0.8, 1.2)
                    if tech == '4G':
                        val = (peso_rel * 100 * (1 - ratio_5g/100) * hora_factor * fluctuacion * dss_4g_factor)
                        val = max(val, gbr_min / 2.2)
                        offset = -0.0003
                    else:
                        val = (peso_rel * 140 * (ratio_5g/100) * hora_factor * fluctuacion * dss_5g_factor)
                        val = max(val, gbr_min / 1.8)
                        offset = 0.0003

                    records.append({
                        "Timestamp": dt, "Site": s, "Tech": tech,
                        "lat": pos['lat'], "lon": pos['lon'] + offset,
                        "Mbps": round(val, 2), "color": color, "Estado_Predictivo": estado
                    })

        df_final = pd.DataFrame(records)

        # --- PANEL PREDICTIVO DE LA TUBERÍA DE DATOS ---
        st.subheader("🤖 Diagnóstico Automático del Espectro (Inferencia del Pipeline)")
        col1, col2, col3 = st.columns(3)
        with col1:
            n_4g = len(site_metrics[site_metrics["Estado_DSS"] == "Dominancia 4G (Bajo Tráfico)"])
            st.metric("Nodos con Dominancia 4G", n_4g)
        with col2:
            n_dss = len(site_metrics[site_metrics["Estado_DSS"] == "DSS Automático Requerido (Tráfico Fluctuante)"])
            st.metric("Nodos con DSS Automático ⚠️", n_dss, delta="Gatillo Activo", delta_color="inverse")
        with col3:
            n_5g = len(site_metrics[site_metrics["Estado_DSS"] == "Dominancia 5G (Alta Carga/Nodos Core)"])
            st.metric("Nodos con Dominancia 5G", n_5g)

        # --- COMPORTAMIENTO TEMPORAL ---
        st.subheader("📊 Comportamiento Temporal según Perfil del Nodo")
        s_pick = st.selectbox("Selecciona una estación para auditar su comportamiento:", list(set(sitios_validos)))
        
        info_sitio = site_metrics[site_metrics[col_site] == s_pick].iloc[0]
        st.info(f"**Resultado de la Inferencia:** Este nodo ha sido clasificado como **{info_sitio['Estado_DSS']}**.")
        
        df_viz = df_final[df_final["Site"] == s_pick].pivot(index="Timestamp", columns="Tech", values="Mbps").reset_index()
        st.line_chart(df_viz.set_index("Timestamp")[["4G", "5G"]])

        # --- MAPA ANALÍTICO 3D ---
        st.subheader("📍 Mapa Analítico 3D: Clasificación de Recursos del Aire")
        st.caption("🔵 Azul = Dominancia 4G | 🟢 Verde = Dominancia 5G | 🟡 Amarillo = Zonas Críticas con DSS Dinámico Automático")
        
        view_state = pdk.ViewState(latitude=df_final.lat.mean(), longitude=df_final.lon.mean(), zoom=12, pitch=50)
        df_mapa_actual = df_final.tail(len(set(sitios_validos)) * 2)
        
        st.pydeck_chart(pdk.Deck(
            layers=[pdk.Layer(
                "ColumnLayer",
                df_mapa_actual,
                get_position=['lon', 'lat'],
                get_elevation='Mbps',
                elevation_scale=100,
                radius=180,
                get_fill_color='color',
                pickable=True
            )],
            initial_view_state=view_state,
            tooltip={"text": "Estación: {Site}\nTecnología: {Tech}\nCarga: {Mbps} Mbps\nClasificación: {Estado_Predictivo}"}
        ))
        
        st.download_button("📥 Descargar Reporte Consolidado", df_final.to_csv(index=False).encode('utf-8'), "Reporte_Pipeline_DSS.csv")
    else:
        st.error("❌ Error de Vínculo: Los nombres en el CSV y el KML son incompatibles.")