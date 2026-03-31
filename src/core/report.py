import os
import io
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy.spatial import cKDTree

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.units import inch

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

# def _plot_heatmap_magnitud_pozos(df_proy, df_pozos):
#     """Grafica los pozos y resalta los que generaron proyecciones según el CSV."""
#     plt.figure(figsize=(5, 4))
#     # Dibujar todos los pozos de la malla como referencia
#     plt.scatter(df_pozos['x'], df_pozos['y'], c='lightgrey', s=15, label='Malla Pozos', alpha=0.6)
    
#     # Pozos con proyecciones asociadas
#     pozos_activos = df_pozos[df_pozos['max_proyeccion'] > 0]
#     if not pozos_activos.empty:
#         scatter = plt.scatter(pozos_activos['x'], pozos_activos['y'], 
#                               c=pozos_activos['max_proyeccion'], 
#                               cmap='YlOrRd', s=45, edgecolors='black', linewidth=0.8)
#         plt.colorbar(scatter, label='Alcance (m)')
    
#     plt.title('Heatmap: Magnitud Máxima por Pozo', fontsize=10, weight='bold')
#     plt.xlabel("Este (m)"); plt.ylabel("Norte (m)")
#     plt.axis('equal')
#     sns.despine()

def _plot_heatmap_magnitud_pozos(df_proy, df_pozos):
    """Grafica los pozos y resalta los que generaron proyecciones según el CSV."""
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
    
    # (plt.xlabel y plt.ylabel fueron eliminados)
    
    plt.axis('equal')
    
    # Quitamos las líneas de los ejes para que quede un gráfico totalmente limpio
    sns.despine(bottom=True, left=True)

def _plot_heatmap_densidad_integrado(df_proy, df_pozos):
    """Densidad de eventos sobre la malla de perforación."""
    plt.figure(figsize=(5, 4))
    plt.scatter(df_pozos['x'], df_pozos['y'], color='black', s=3, alpha=0.5)
    
    if len(df_proy) > 1:
        sns.kdeplot(x=df_proy['origen_x'], y=df_proy['origen_y'], 
                    cmap='Reds', fill=True, thresh=0.05, alpha=0.7)
    
    plt.title('Heatmap: Concentración de Eventos', fontsize=10, weight='bold')
    plt.axis('equal')
    plt.axis('off')

# ==========================================
# 2. DISEÑO CORPORATIVO
# ==========================================

def _draw_corporate_decorations(canvas, doc):
    width, height = letter
    canvas.saveState()
    # Franja Superior
    canvas.setFillColor(colors.HexColor("#D3D3D3"))
    canvas.rect(0, height - 70, width, 70, stroke=0, fill=1)
    canvas.setFont("Helvetica-Bold", 8)
    canvas.setFillColor(colors.black)
    canvas.drawString(40, height - 25, "ANÁLISIS DETOVISION ®")
    canvas.setFillColor(colors.red)
    canvas.drawString(40, height - 40, "ENAEX MINING TECHNICAL SOLUTIONS")
    canvas.drawRightString(width - 40, height - 40, "Enaex")
    # Franja Inferior
    canvas.setFillColor(colors.HexColor("#4F5B66"))
    canvas.rect(0, 0, width, 30, stroke=0, fill=1)
    canvas.restoreState()

# ==========================================
# 3. LÓGICA DE PROCESAMIENTO INTEGRADO
# ==========================================


def procesar_datos_final(json_path, csv_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data_json = json.load(f)
        
        df_pozos = pd.read_csv(csv_path)
        
        # NORMALIZACIÓN DE COLUMNAS: Forzamos minúsculas para evitar errores de 'X' vs 'x'
        df_pozos.columns = [c.lower().strip() for c in df_pozos.columns]
        
        if 'x' not in df_pozos.columns or 'y' not in df_pozos.columns:
            raise ValueError(f"El CSV debe contener columnas 'x' e 'y'. Encontradas: {list(df_pozos.columns)}")

        df_pozos['max_proyeccion'] = 0.0

        procesados = []
        for item in data_json:
            # Validación de estructura de coordenadas
            coords = item.get('coords_mina_inicio', [0, 0])
            ox = coords[0] if len(coords) >= 1 else 0
            oy = coords[1] if len(coords) >= 2 else 0
            
            procesados.append({
                'origen_x': ox, 'origen_y': oy,
                'distancia_metros': item.get('distancia_metros', 0),
                'excede_filmacion': (item.get('clasificacion', '') == 'Fuera de vista')
            })
        
        df_proy = pd.DataFrame(procesados)

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

def generar_pdf_job(json_path: str, csv_path: str, output_pdf: str, radio_equipos: float = 250.0):
    """Retorna True si tuvo éxito, False si falló."""
    try:
        if not os.path.exists(json_path): raise FileNotFoundError(f"No existe JSON: {json_path}")
        if not os.path.exists(csv_path): raise FileNotFoundError(f"No existe CSV: {csv_path}")

        df_proy, df_pozos = procesar_datos_final(json_path, csv_path)
        
        doc = SimpleDocTemplate(output_pdf, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=90, bottomMargin=60)
        styles = getSampleStyleSheet()
        story = []

        # Título y Contenido (Lógica anterior...)
        style_title = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=16, alignment=1)
        story.append(Paragraph('Reporte Técnico Detovision®', style_title))
        story.append(Spacer(1, 0.2*inch))

        # --- TABLA Y GRÁFICOS ---
        # (Aquí va tu lógica de buffers e imágenes...)
        # Asegúrate de que las funciones de plot reciban los dataframes procesados
        buf_hist = _create_plot_buffer(_plot_hist_cumulative, df_proy['distancia_metros'], 'Distribución', 'Metros')
        buf_magnitud = _create_plot_buffer(_plot_heatmap_magnitud_pozos, df_proy, df_pozos)
        buf_densidad = _create_plot_buffer(_plot_heatmap_densidad_integrado, df_proy, df_pozos)

        graficos_table = Table([[Image(buf_magnitud, 3.4*inch, 2.8*inch), Image(buf_densidad, 3.4*inch, 2.8*inch)]])
        story.append(graficos_table)
        story.append(Spacer(1, 0.2*inch))
        story.append(Image(buf_hist, width=4*inch, height=2.5*inch))

        doc.build(story, onFirstPage=_draw_corporate_decorations, onLaterPages=_draw_corporate_decorations)
        
        # VERIFICACIÓN FINAL: ¿Se creó el archivo?
        if os.path.exists(output_pdf):
            return True
        return False

    except Exception as e:
        print(f"CRITICAL ERROR generando PDF: {str(e)}")
        # Importante: si falla, lanzamos el error para que el service.py lo capture
        raise e