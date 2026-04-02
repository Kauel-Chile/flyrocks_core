import os
import io
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy.spatial import cKDTree

# ReportLab imports
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

# ==========================================
# 1. FUNCIONES DE APOYO Y GRÁFICOS
# ==========================================

def _create_plot_buffer(plot_func, *args, **kwargs):
    buf = io.BytesIO()
    plot_func(*args, **kwargs)
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    plt.close()
    buf.seek(0)
    return buf

def _plot_hist_cumulative(data_series, title, xlabel):
    fig, ax1 = plt.subplots(figsize=(6, 3.5))
    if data_series.empty:
        ax1.text(0.5, 0.5, "Sin datos", ha='center')
        return

    n, bins, patches = ax1.hist(data_series, bins=15, color='#5DADE2', edgecolor='white')
    ax1.set_xlabel(xlabel, fontsize=9)
    ax1.set_ylabel('Frecuencia', fontsize=9)
    
    ax2 = ax1.twinx()
    cumulative = np.cumsum(n)
    if cumulative[-1] > 0:
        ax2.plot(bins[1:], (cumulative / cumulative[-1]) * 100, color='#E67E22', marker='o', linewidth=2)
    ax2.set_ylabel('Acumulado (%)', fontsize=9)
    plt.title(title, fontsize=10, weight='bold')

def _plot_heatmap_magnitud_pozos(df_proy, df_pozos):
    """Grafica los pozos y resalta los que generaron proyecciones según el CSV, sin etiquetas en ejes."""
    plt.figure(figsize=(5, 4))
    
    # Dibujar todos los pozos de la malla como referencia
    plt.scatter(df_pozos['x'], df_pozos['y'], c='lightgrey', s=15, label='Malla Pozos', alpha=0.6)
    
    # Pozos con proyecciones asociadas
    pozos_activos = df_pozos[df_pozos['max_proyeccion'] > 0]
    if not pozos_activos.empty:
        scatter = plt.scatter(pozos_activos['x'], pozos_activos['y'], 
                              c=pozos_activos['max_proyeccion'], 
                              cmap='YlOrRd', s=45, edgecolors='black', linewidth=0.8)
        
        # Guardamos la referencia del colorbar, sin etiqueta (label)
        cbar = plt.colorbar(scatter)
        # Quitamos los números (ticks) del colorbar
        cbar.set_ticks([])
    
    plt.title('Heatmap: Magnitud Máxima por Pozo', fontsize=10, weight='bold')
    
    # Quitamos los números de los ejes X e Y
    plt.xticks([])
    plt.yticks([])
    plt.axis('equal')
    
    # Quitamos las líneas de los ejes
    sns.despine(bottom=True, left=True)

def _plot_heatmap_densidad_integrado(df_proy, df_pozos):
    """Densidad de eventos sobre la malla de perforación."""
    plt.figure(figsize=(5, 4))
    plt.scatter(df_pozos['x'], df_pozos['y'], color='black', s=3, alpha=0.5)
    
    if len(df_proy) > 1:
        sns.kdeplot(x=df_proy['origen_x'], y=df_proy['origen_y'], 
                    cmap='Reds', fill=True, thresh=0.05, alpha=0.7)
    
    plt.title('Heatmap: Concentración de Eventos', fontsize=10, weight='bold')
    plt.xticks([])
    plt.yticks([])
    plt.axis('equal')
    sns.despine(bottom=True, left=True)

# ==========================================
# 2. DISEÑO CORPORATIVO (HEADER/FOOTER)
# ==========================================

def _draw_corporate_decorations(canvas, doc):
    width, height = letter
    canvas.saveState()
    
    # --- HEADER ---
    # Rectángulo Gris Claro (Izquierda)
    canvas.setFillColor(HexColor("#DCDCDC"))
    canvas.rect(0, height - 80, width * 0.55, 80, stroke=0, fill=1)
    
    # Rectángulo Gris Oscuro (Derecha)
    canvas.setFillColor(HexColor("#4F5B66"))
    canvas.rect(width * 0.55, height - 80, width * 0.45, 80, stroke=0, fill=1)
    
    # Texto Header (Datos del Proyecto)
    canvas.setFillColor(HexColor("#4F5B66"))
    canvas.setFont("Helvetica-Bold", 8)
    canvas.drawString(40, height - 25, "ANÁLISIS DETOVISION")
    canvas.setFillColor(HexColor("#E74C3C")) # Rojo Enaex
    canvas.drawString(40, height - 38, "PROYECTO - (GENERADO AUTOMÁTICAMENTE)")
    canvas.setFillColor(HexColor("#4F5B66"))
    canvas.drawString(40, height - 51, "ENAEX MINING TECHNICAL SOLUTIONS - EMTS")
    
    # --- FOOTER ---
    # Barra Inferior Gris Oscura
    canvas.setFillColor(HexColor("#4F5B66"))
    canvas.rect(0, 0, width, 35, stroke=0, fill=1)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(40, 15, "www.enaex.com")
    
    # Paginación
    canvas.drawCentredString(width / 2.0, 15, f"Página {doc.page}")
    
    canvas.restoreState()

# ==========================================
# 3. LÓGICA DE PROCESAMIENTO INTEGRADO
# ==========================================

def procesar_datos_final(json_path, csv_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data_json = json.load(f)
        
        df_pozos = pd.read_csv(csv_path)
        
        # NORMALIZACIÓN DE COLUMNAS: Forzamos minúsculas
        df_pozos.columns = [c.lower().strip() for c in df_pozos.columns]
        
        if 'x' not in df_pozos.columns or 'y' not in df_pozos.columns:
            raise ValueError(f"El CSV debe contener columnas 'x' e 'y'. Encontradas: {list(df_pozos.columns)}")

        df_pozos['max_proyeccion'] = 0.0

        procesados = []
        for item in data_json:
            coords = item.get('coords_mina_inicio', [0, 0])
            ox = coords[0] if len(coords) >= 1 else 0
            oy = coords[1] if len(coords) >= 2 else 0
            
            procesados.append({
                'origen_x': ox, 'origen_y': oy,
                'distancia_metros': item.get('distancia_metros', 0),
                'excede_filmacion': (item.get('clasificacion', '') == 'Fuera de vista')
            })
        
        df_proy = pd.DataFrame(procesados)

        # Asociación espacial de eventos a los pozos más cercanos
        if not df_proy.empty and not df_pozos.empty:
            tree = cKDTree(df_pozos[['x', 'y']].values)
            _, indices = tree.query(df_proy[['origen_x', 'origen_y']].values)
            
            for i, idx_pozo in enumerate(indices):
                dist_roca = df_proy.iloc[i]['distancia_metros']
                if dist_roca > df_pozos.at[idx_pozo, 'max_proyeccion']:
                    df_pozos.at[idx_pozo, 'max_proyeccion'] = dist_roca
                    
        return df_proy, df_pozos
    except Exception as e:
        print(f"Error en procesar_datos_final: {e}")
        raise e

# ==========================================
# 4. GENERADOR PRINCIPAL DEL REPORTE
# ==========================================

def generar_pdf_job(json_path: str, csv_path: str, output_pdf: str, radio_equipos: float = 250.0):
    """Retorna True si tuvo éxito, False si falló."""
    try:
        if not os.path.exists(json_path): raise FileNotFoundError(f"No existe JSON: {json_path}")
        if not os.path.exists(csv_path): raise FileNotFoundError(f"No existe CSV: {csv_path}")

        # 1. Procesamiento de datos
        df_proy, df_pozos = procesar_datos_final(json_path, csv_path)
        
        # 2. Variables Dinámicas para el texto
        fecha_hoy = datetime.now().strftime("%d/%m/%Y")
        total_proy = len(df_proy)
        max_dist = df_proy['distancia_metros'].max() if not df_proy.empty else 0
        exceden_radio = len(df_proy[df_proy['distancia_metros'] > radio_equipos]) if not df_proy.empty else 0
        pozos_involucrados = len(df_pozos[df_pozos['max_proyeccion'] > 0])
        
        # 3. Configuración del Documento
        doc = SimpleDocTemplate(output_pdf, pagesize=letter, 
                                rightMargin=40, leftMargin=40, 
                                topMargin=100, bottomMargin=50)
        styles = getSampleStyleSheet()
        
        # Estilos personalizados
        style_title = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=16, alignment=TA_CENTER, textColor=HexColor("#333333"))
        style_heading = ParagraphStyle('Heading', parent=styles['Heading2'], fontSize=14, textColor=HexColor("#4F5B66"), spaceAfter=6)
        style_normal = ParagraphStyle('Normal_Just', parent=styles['Normal'], fontSize=10, alignment=TA_JUSTIFY, spaceAfter=8, leading=12)
        style_caption = ParagraphStyle('Caption', parent=styles['Normal'], fontSize=8, alignment=TA_CENTER, fontName='Helvetica-Oblique', textColor=HexColor("#555555"), spaceAfter=12)
        
        story = []

        # # --- Título Principal ---
        # story.append(Paragraph(f'<b>Análisis Detovision® – <font color="red">Reporte Dinámico</font></b>', style_title))
        # story.append(Spacer(1, 0.2*inch))

        # --- Generación de Gráficos (Buffers) ---
        buf_hist = _create_plot_buffer(_plot_hist_cumulative, df_proy['distancia_metros'], 'Histograma de distancia máxima', 'Distancia (m)')
        buf_densidad = _create_plot_buffer(_plot_heatmap_densidad_integrado, df_proy, df_pozos)
        buf_magnitud = _create_plot_buffer(_plot_heatmap_magnitud_pozos, df_proy, df_pozos)

        # Configuramos el ancho de las columnas
        page_width = letter[0] - 80 # Ancho total menos márgenes
        col_width = (page_width / 2.0) - 10 # Restamos 10 puntos de separación
        img_width = col_width - 10 # La imagen un poco más chica que la columna
        
        # --- COLUMNA IZQUIERDA ---
        left_col = []
        left_col.append(Paragraph('<b>Introducción</b>', style_heading))
        left_col.append(Paragraph(f'La presente nota técnica tiene por propósito determinar distancias de proyección de roca con herramienta detovision en tronadura procesada el día <font color="red">{fecha_hoy}</font>.', style_normal))
        
        left_col.append(Paragraph('<b>Análisis</b>', style_heading))
        left_col.append(Paragraph(f'En la Figura 1 se puede observar la concentración de eventos asociados a la tronadura. Una vez identificadas las proyecciones por el software, se validaron un total de <b>{total_proy} fragmentos</b>.', style_normal))
        
        # Insertamos Gráfico 1 (Mapa de densidad)
        left_col.append(Image(buf_densidad, width=img_width, height=img_width * 0.75))
        left_col.append(Paragraph('Figura 1. Concentración de eventos detectados', style_caption))
        
        left_col.append(Paragraph(f'De las trayectorias analizadas, se determinó un radio de seguridad de la mina para equipos de <font color="red"><b>{radio_equipos}m</b></font>. Se observó un alcance máximo general de {max_dist:.1f}m.', style_normal))
        
        if exceden_radio > 0:
            left_col.append(Paragraph(f'<b>Atención:</b> En el registro se observan <font color="red">{exceden_radio} fragmentos</font> que escapan del límite de seguridad establecido para los equipos.', style_normal))
        else:
            left_col.append(Paragraph(f'Acorde a los registros, ningún fragmento documentado superó el límite de seguridad de {radio_equipos}m.', style_normal))


        # --- COLUMNA DERECHA ---
        right_col = []
        # Insertamos Gráfico 2 (Histograma)
        right_col.append(Image(buf_hist, width=img_width, height=img_width * 0.65))
        right_col.append(Paragraph('Figura 2. Distancias máximas', style_caption))
        
        right_col.append(Paragraph('La Figura 2 muestra el histograma que se construye en función a los fragmentos identificados y presenta la distribución de los datos acumulados de las proyecciones.', style_normal))
        
        right_col.append(Paragraph('En la Figura 3 se muestra el mapa de calor que asocia las magnitudes máximas de vuelo directamente a los pozos de la malla de perforación.', style_normal))
        
        # Insertamos Gráfico 3 (Heatmap de pozos)
        right_col.append(Image(buf_magnitud, width=img_width, height=img_width * 0.75))
        right_col.append(Paragraph('Figura 3. HeatMap magnitud por pozo', style_caption))
        
        right_col.append(Paragraph('<b>Observaciones</b>', style_heading))
        obs_texto = f"Durante el análisis de la tronadura, se detectó la concentración de proyecciones sobre las zonas ilustradas. Un total de {pozos_involucrados} pozos de la malla registraron eventos de proyección detectables asociados a su proximidad espacial."
        right_col.append(Paragraph(obs_texto, style_normal))

        # --- ENSAMBLAJE DE LAS COLUMNAS ---
        layout_table = Table([[left_col, right_col]], colWidths=[col_width, col_width])
        layout_table.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'TOP'), 
            ('LEFTPADDING', (0,0), (-1,-1), 0),
            ('RIGHTPADDING', (0,0), (-1,-1), 0),
            ('BOTTOMPADDING', (0,0), (-1,-1), 0),
            ('TOPPADDING', (0,0), (-1,-1), 0),
        ]))
        
        story.append(layout_table)

        # 4. Construir Documento
        doc.build(story, onFirstPage=_draw_corporate_decorations, onLaterPages=_draw_corporate_decorations)
        
        if os.path.exists(output_pdf):
            return True
        return False

    except Exception as e:
        print(f"CRITICAL ERROR generando PDF: {str(e)}")
        raise e