import os
import json
import shutil
import asyncio
import sys
import time
import uuid
from pathlib import Path
import logging

# Nadie configuraba el logging: el root queda en WARNING y los logger.info del
# pipeline se descartan en silencio (el filtro de humo, el nodo mas caro, era el
# unico del que no se veia una linea).
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
    force=True,          # uvicorn ya instalo los suyos; este manda
)

# La consola de Windows en cp1252 no sabe escribir los emojis de los print de
# progreso: el print revienta con UnicodeEncodeError y mata la corrida entera.
for _flujo in (sys.stdout, sys.stderr):
    try:
        _flujo.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass  # flujo redirigido o sin soporte: no vale la pena tumbar el arranque

from fastapi import FastAPI, BackgroundTasks, WebSocket, WebSocketDisconnect, File, UploadFile, Form, HTTPException
from pydantic import BaseModel
from sqlmodel import SQLModel, Session
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from datetime import datetime, timedelta

from utils.database import engine, Job, migrar
from utils.services import run_pipeline_task, TEMP_VIDEOS
from utils import malla as malla_utils


# Todo bajo DATA_DIR (el volumen), pero la URL publica sigue siendo /temp_videos
# para no romper al frontend.
os.makedirs(TEMP_VIDEOS, exist_ok=True)

# Horas que se conservan los analisis. 0 = para siempre, que es el DEFECTO a
# proposito: el core guarda el trabajo del cliente, y borrarle un analisis sin
# que lo haya pedido es peor que quedarse sin disco. Se enciende poniendo un
# numero en el compose, igual que el blast detector (que si tiene 24 h porque lo
# suyo son videos crudos, no resultados).
RETENCION_HORAS = float(os.getenv("RETENCION_HORAS", "0") or 0)


def purgar_analisis_viejos():
    """Borra los analisis mas viejos que RETENCION_HORAS, con sus archivos."""
    if RETENCION_HORAS <= 0:
        return
    corte = datetime.utcnow() - timedelta(hours=RETENCION_HORAS)
    with Session(engine) as session:
        viejos = [j for j in session.query(Job).all()
                  if j.creado_en and j.creado_en < corte and not j.is_running]
        for j in viejos:
            carpeta = ((j.entrada or {}).get("artefactos") or {}).get("carpeta")
            if carpeta:
                destino = (Path(TEMP_VIDEOS) / carpeta).resolve()
                if destino.parent == Path(TEMP_VIDEOS).resolve() and destino.is_dir():
                    shutil.rmtree(destino, ignore_errors=True)
            session.delete(j)
        if viejos:
            session.commit()
            print(f"[retencion] {len(viejos)} analisis de mas de "
                  f"{RETENCION_HORAS} h eliminados")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Inicializando recursos de la aplicación...")
    SQLModel.metadata.create_all(engine)
    migrar()   # columnas nuevas sobre una base que ya existe
    purgar_analisis_viejos()
    yield 
    print("Apagando la aplicación y liberando recursos...")
    engine.dispose()

app = FastAPI(title="API de Análisis Flyrocks", lifespan=lifespan)

class _EstaticoSinCache(StaticFiles):
    """Pide revalidar siempre: sin esta cabecera el navegador puede mostrar el
    artefacto cacheado de otro analisis, que es como se veia el bug."""

    def file_response(self, *args, **kwargs):
        resp = super().file_response(*args, **kwargs)
        resp.headers.setdefault("Cache-Control", "no-cache")
        return resp


app.mount("/temp_videos", _EstaticoSinCache(directory=TEMP_VIDEOS), name="temp_videos")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En desarrollo permitimos todo. En prod, pones la URL de tu front
    allow_credentials=True,
    allow_methods=["*"],  # Permite POST, GET, OPTIONS, etc.
    allow_headers=["*"],
)

# El pipeline se PAUSA aca (fase 1 terminada) y espera que el usuario elija el
# percentil en el paso 4. El job queda con is_running=True a proposito: es lo
# que mantiene vivo el WebSocket de progreso mientras el usuario decide.
ESPERANDO_PERCENTIL = "ESPERANDO_PERCENTIL_USUARIO"

class ResumeRequest(BaseModel):
    percentile: float

def _guardar_artefactos(carpeta: Path, job_id: str, nombre_video: str, ancla):
    """Extrae el frame de referencia y devuelve las rutas RELATIVAS a
    /temp_videos, que es como la vista arma sus URLs."""
    import cv2

    ruta_video = carpeta / nombre_video
    destino = carpeta / "frame.jpg"
    try:
        cap = cv2.VideoCapture(str(ruta_video))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        # Tres frames antes del primer tiro; si no hay ancla, el primero del clip
        # (que tambien es pre-tronadura, porque el corte empieza antes).
        objetivo = max(0, (ancla or 3) - 3)
        if total and objetivo >= total:
            objetivo = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, objetivo)
        ok, frame = cap.read()
        if not ok:                      # algunos codecs no aceptan el salto
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        cap.release()
        if ok:
            cv2.imwrite(str(destino), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            print(f"[frame] referencia del frame {objetivo} -> {destino.name}")
    except Exception as e:              # no vale la pena tumbar el analisis
        print(f"[frame] no se pudo extraer el frame de referencia: {e}")

    web = _derivar_video_web(carpeta, nombre_video)

    return {
        "carpeta": job_id,
        "mascara": f"{job_id}/mascara_cambios.png",
        "frame": f"{job_id}/frame.jpg" if destino.exists() else None,
        # El derivado reproducible si se pudo lanzar; si no, el original, que al
        # menos sirve para descargarlo aunque el navegador no lo pinte.
        "video": f"{job_id}/{web}" if web else f"{job_id}/{nombre_video}",
        "video_original": f"{job_id}/{nombre_video}",
    }


# Códecs que un navegador reproduce de verdad. El resto hay que convertirlo.
_CODECS_WEB = ("avc1", "h264")


def _derivar_video_web(carpeta: Path, nombre_video: str):
    """Deja un derivado H.264 del clip para que el fondo de video se vea.

    El recorte llega de OpenCV con fourcc mp4v (MPEG-4 parte 2): ningun navegador
    lo decodifica y el <video> queda en negro con MEDIA_ERR_SRC_NOT_SUPPORTED.
    Es un DERIVADO y no una conversion del original porque reescribir el clip
    invalidaria la cache de todos los nodos del pipeline. Corre en un hilo
    aparte: son ~26 s sobre un 4K y el POST tiene que responder al toque.
    """
    import shutil as _sh
    import subprocess
    import threading

    origen = carpeta / nombre_video
    if not origen.exists():
        return None

    # Si ya viene en H.264 no se toca: transcodificar de nuevo solo perderia
    # calidad y tiempo.
    try:
        import cv2
        cap = cv2.VideoCapture(str(origen))
        cc = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
        cap.release()
        fourcc = "".join(chr((cc >> 8 * i) & 0xFF) for i in range(4)).lower()
        if fourcc in _CODECS_WEB:
            print(f"[video] {nombre_video} ya es {fourcc}: no hace falta derivado")
            return None
    except Exception as e:
        fourcc = "?"
        print(f"[video] no se pudo leer el codec ({e}); se genera derivado igual")

    if not _sh.which("ffmpeg"):
        print("[video] sin ffmpeg: el fondo de video no se va a poder reproducir")
        return None

    salida = "video_web.mp4"
    # Se escribe a un temporal y se renombra al final. Si no, la vista puede
    # pedir el archivo a medio escribir y fallar igual que antes, pero ahora sin
    # que se entienda por que.
    tmp = carpeta / "video_web.parcial.mp4"
    final = carpeta / salida

    def convertir():
        cmd = [
            "ffmpeg", "-v", "error", "-y", "-i", str(origen),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
            # GOP corto: la vista salta frame a frame y con keyframes cada 250
            # cuadros cada salto obliga a decodificar medio segundo de video.
            "-g", "15",
            "-pix_fmt", "yuv420p",
            # El moov al principio. OpenCV lo deja al final, y asi el navegador
            # tiene que bajarse el archivo entero antes de pintar el primer
            # cuadro (92 MB por un frame).
            "-movflags", "+faststart",
            "-an", str(tmp),
        ]
        try:
            t0 = time.time()
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            if r.returncode == 0 and tmp.exists():
                tmp.replace(final)
                mb = final.stat().st_size / 1e6
                print(f"[video] derivado H.264 listo en {time.time()-t0:.0f}s "
                      f"({mb:.0f} MB, desde {fourcc})")
            else:
                print(f"[video] ffmpeg fallo ({r.returncode}): "
                      f"{(r.stderr or '').strip()[:300]}")
                tmp.unlink(missing_ok=True)
        except Exception as e:
            print(f"[video] no se pudo derivar el clip: {e}")
            tmp.unlink(missing_ok=True)

    threading.Thread(target=convertir, daemon=True).start()
    print(f"[video] {nombre_video} viene en {fourcc}: derivando H.264 en segundo plano")
    return salida


def _fps_de(video_path: str):
    """FPS del video, o None. Hace falta para pasar el tiempo de detonacion (ms
    en el CSV) a frame, que es la unidad de las trayectorias."""
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return round(float(fps), 3) if fps and fps > 0 else None
    except Exception:
        return None


# --- ENDPOINT PARA DISPARAR EL ANÁLISIS ---
# Cambiamos la ruta a /api/analyze para que haga match con el fetch del JS
@app.post("/api/analyze")
async def start_analysis(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    origin_zone: str = Form(...),
    expected_projection_zone: str = Form(...),
    h_matrix: str = Form(...),
    percentile: float = Form(..., ge=0.0, le=100.0),
    sigma: float = Form(..., ge=0.0, le=1.0),
    esp: float = Form(..., ge=1.0, le=7.0),
    # CSV de secuencia (Label, X, Y, Z, DetonatingTime). OPCIONAL: sin el, el
    # analisis corre igual pero el job queda sin malla y sin asociacion al tiro.
    detonation_sequence: UploadFile = File(None),
    # El ancla temporal, en frames del video ORIGINAL. Ya existia aguas arriba
    # (paso 2 del wizard) y hasta ahora se tiraba.
    frame_detonacion: int = Form(None),   # lo que detectó el blast detector
    frame_inicio_corte: int = Form(None), # dónde cortó el usuario
):
    # 1. Parsear y validar los strings JSON que vienen del form
    try:
        origin_zone_parsed = json.loads(origin_zone)
        expected_zone_parsed = json.loads(expected_projection_zone)
        h_matrix_parsed = json.loads(h_matrix)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Los parámetros de zonas o matriz deben ser JSON válidos.")

    # 2. UNA CARPETA POR ANALISIS. Antes todo iba junto con nombres fijos: la
    # mascara de cambios era UNA sola para toda la app y cada analisis pisaba la
    # del anterior — abrir un job viejo mostraba la mascara de otro, sin error.
    job_id = str(uuid.uuid4())
    carpeta = Path(TEMP_VIDEOS) / job_id
    carpeta.mkdir(parents=True, exist_ok=True)
    nombre_video = Path(video.filename or "video").name or "video"

    # El wizard sube el recorte como `video`, SIN extension, y el estatico
    # adivina el tipo por ella: sin extension lo sirve como text/plain.
    if not Path(nombre_video).suffix:
        nombre_video += ".mp4"
    video_path = str(carpeta / nombre_video)
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    # 3. Se guarda CON QUE se corrio: sin homografia ni zonas, ninguna vista
    # puede dibujar la malla ni asociar al tiro.
    entrada = {
        "video": video.filename,
        "h_matrix": h_matrix_parsed,
        "origin_zone": origin_zone_parsed,
        "expected_projection_zone": expected_zone_parsed,
        "parametros": {"percentile": percentile, "sigma": sigma, "esp": esp},
    }


    # Se guardan LOS TRES: el ancla es lo que usa la vista, los crudos permiten
    # recalcularla si manana cambia el criterio.
    if frame_detonacion is not None and frame_inicio_corte is not None:
        entrada["recorte"] = {
            "frame_detonacion": frame_detonacion,
            "frame_inicio_corte": frame_inicio_corte,
            "ancla_frames": frame_detonacion - frame_inicio_corte,
        }

        print(f"[ancla] {frame_detonacion} - {frame_inicio_corte} = "
              f"{frame_detonacion - frame_inicio_corte} frames")


    entrada["artefactos"] = _guardar_artefactos(
        carpeta, job_id, nombre_video,
        entrada.get("recorte", {}).get("ancla_frames"))


    if detonation_sequence is not None and detonation_sequence.filename:
        crudo = await detonation_sequence.read()
        entrada["secuencia"] = {
            "archivo": detonation_sequence.filename,
            "csv": crudo.decode("utf-8-sig", errors="replace"),
        }
        try:
            entrada["malla"] = malla_utils.desde_csv(
                crudo, h_matrix_parsed, fps=_fps_de(video_path)
            )
            print(f"[malla] {entrada['malla']['meta']['n_pozos']} pozos proyectados")
        except Exception as e:
            entrada["malla_error"] = str(e)
            print(f"[malla] no se pudo procesar el CSV: {e}")
    with Session(engine) as session:
        new_job = Job(id=job_id, status="Iniciando...", progress=0, entrada=entrada)
        session.add(new_job)
        session.commit()
        session.refresh(new_job)

    background_tasks.add_task(
        run_pipeline_task, 
        new_job.id, 
        video_path, 
        origin_zone_parsed, 
        expected_zone_parsed, 
        h_matrix_parsed,
        percentile,
        sigma,
        esp,    
        output_filename="voladura_analisis.mp4"  
    )
    
    return {"job_id": new_job.id, "mensaje": "Análisis encolado en segundo plano"}

# --- ENDPOINT PARA REANUDAR (O REBOBINAR) TRAS LA SELECCIÓN DE PERCENTIL ---
@app.post("/api/resume/{job_id}")
async def resume_analysis(
    job_id: str,
    body: ResumeRequest,
    background_tasks: BackgroundTasks
):
    with Session(engine) as session:
        job = session.get(Job, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Análisis no encontrado")
        
        # ELIMINAMOS LA RESTRICCIÓN RÍGIDA
        # Ya no bloqueamos si el estado es distinto a "ESPERANDO_PERCENTIL_USUARIO".
        # Si el job ya terminó (is_running=False), lo "resucitamos".
        
        entrada = job.entrada or {}
        parametros = entrada.get("parametros", {})
        parametros["percentile"] = body.percentile
        entrada["parametros"] = parametros
        
        job.entrada = entrada
        job.status = "Reanudando pipeline con nuevo percentil..."
        job.is_running = True   # Resucita el job para el WebSocket
        job.progress = 18       # Retrocede la barra de progreso
        job.result_file_path = None
        job.error_message = None
        job.json_data = None    # Borramos los resultados viejos
        # Y con ellos la edicion manual: los track_id se reasignan desde cero al
        # recalcular, asi que el avance de la pasada anterior apunta a rocas que
        # ya no son esas. Rebobinar es volver atras en el wizard, y lo de adelante
        # se pierde. El archivo que el usuario descargo sigue siendo su respaldo.
        job.avance = None
        
        session.add(job)
        session.commit()

        video_path = os.path.join(TEMP_VIDEOS, job_id, f"{entrada.get('video', 'video.mp4')}")
        if not os.path.exists(video_path):
            video_path = os.path.join(TEMP_VIDEOS, job_id, "video.mp4")

        background_tasks.add_task(
            run_pipeline_task,
            job_id,
            video_path,
            entrada.get("origin_zone", []),
            entrada.get("expected_projection_zone", []),
            entrada.get("h_matrix", []),
            body.percentile,
            parametros.get("sigma", 0.5),
            parametros.get("esp", 5.0)
        )

    return {"job_id": job_id, "mensaje": f"Pipeline reanudado/rebobinado con percentil {body.percentile}%"}
# --- WEBSOCKET PARA NOTIFICAR EL AVANCE ---
@app.websocket("/ws/progress/{job_id}")
async def websocket_job_status(websocket: WebSocket, job_id: str):
    await websocket.accept()
    try:
        while True:
            job_data = None
            
            # --- BLOQUE 1: Leer la Base de Datos con cuidado ---
            try:
                with Session(engine) as session:
                    job = session.get(Job, job_id)
                    if job:
                        job_data = {
                            "id": job.id,
                            "status": job.status,
                            "percentage": job.progress,
                            "is_running": job.is_running,
                            "result_file_path": job.result_file_path,
                            "error_message": job.error_message,
                            "has_report": False
                        }
            except Exception as db_error:
                # Solo atrapamos errores de SQLite aquí
                print(f"⏳ Base de datos ocupada. Reintentando...")
                await asyncio.sleep(1)
                continue  # Volvemos al inicio del while

            # Si el job_id no existe en la base de datos
            if not job_data:
                await websocket.send_json({"error": "Job no encontrado"})
                break
            
            # --- BLOQUE 2: Enviar los datos al Frontend ---
            # Si el frontend se desconectó, esto lanzará un error que romperá el while
            await websocket.send_json(job_data)

            # Si el proceso terminó con éxito o error, cerramos el bucle
            if not job_data["is_running"]:
                break
                
            # Esperamos 1 segundo antes de la próxima actualización
            await asyncio.sleep(1)
            
        # Si salimos del bucle limpiamente, cerramos la conexión
        await websocket.close()
        
    except WebSocketDisconnect:
        print(f"🔌 Cliente desconectado normalmente del job {job_id}")
    except RuntimeError as e:
        print(f"🔌 Conexión cerrada inesperadamente: {str(e)}")
    except Exception as e:
        print(f"❌ Error inesperado en el WebSocket: {str(e)}")
        
@app.get("/api/jobs")
def list_jobs(limite: int = 50):

    from sqlalchemy import text

    with Session(engine) as session:
        filas = session.execute(text("""
            SELECT id, creado_en, status, is_running,
                   json_extract(entrada, '$.video')              AS video,
                   json_extract(entrada, '$.artefactos.carpeta') AS carpeta,
                   json_extract(entrada, '$.h_matrix')           AS h_matrix,
                   -- El resumen del avance viaja en el propio JSON, y se lee
                   -- con json_extract para NO traer el avance entero: son
                   -- megas por job y esta lista se pide solo para elegir.
                   json_extract(avance, '$._meta.guardado_en')   AS avance_en,
                   json_extract(avance, '$._meta.aprobadas')     AS avance_aprobadas,
                   CASE WHEN json_data IS NULL THEN 0
                        ELSE (SELECT count(*) FROM json_each(job.json_data)) END AS trayectorias
            FROM job
            ORDER BY creado_en DESC, rowid DESC
            LIMIT :limite
        """), {"limite": limite}).mappings().all()

    salida = []
    for f in filas:
        d = dict(f)
        # ABRIBLE: si la vista va a poder reconstruir el analisis o no.
        #
        # Sin esto, un job de una version anterior aparecia en la lista igual que
        # los demas y reventaba al abrirlo — o peor: sin `artefactos` caia al
        # nombre global `mascara_cambios.png` y mostraba LA MASCARA DE OTRO
        # ANALISIS, sin un solo error, solo una imagen que no corresponde. Es
        # mejor no ofrecerlo y decir por que.
        carpeta = d.pop("carpeta", None)
        falta = []
        if not d.pop("h_matrix", None): falta.append("la calibración")
        if not carpeta:                 falta.append("sus imágenes")

        # DONDE RETOMAR. Un analisis no siempre quedo terminado: desde que el
        # pipeline se parte en dos, puede estar esperando que el usuario elija
        # el percentil. Esos NO se pueden esconder de la lista —es la unica
        # puerta de vuelta, y el trabajo caro ya se hizo— pero tampoco abrirse
        # en la vista final, porque todavia no hay trayectorias.
        pausado = d["status"] == ESPERANDO_PERCENTIL
        # Un analisis que reventó tampoco lleva a ninguna parte: no tiene
        # trayectorias, asi que abrirlo seria una vista vacia. Se muestra igual
        # —ocupa disco y hay que poder borrarlo— pero no se ofrece.
        fallado = str(d["status"] or "").startswith("Error")
        if fallado:
            d["retomar_en"] = None
        elif pausado:
            d["retomar_en"] = "percentil"
        elif d["is_running"]:
            d["retomar_en"] = None          # corriendo: no hay nada que abrir
        else:
            d["retomar_en"] = "edicion"

        d["abrible"] = not falta and not pausado and not fallado and not d["is_running"]
        if fallado:
            d["motivo"] = "Terminó con error"
        elif falta:
            d["motivo"] = ("Análisis de una versión anterior: no guardó "
                           + " ni ".join(falta))
        elif pausado:
            d["motivo"] = "Quedó esperando que elijas el corte de ruido"
        elif d["is_running"]:
            d["motivo"] = "Procesando…"
        else:
            d["motivo"] = None

        # Lo que ocupa en disco, para que se vea que esta llenando el equipo.
        d["bytes"] = _peso_carpeta(Path(TEMP_VIDEOS) / carpeta) if carpeta else 0
        salida.append(d)
    return salida


def _peso_carpeta(ruta: Path) -> int:
    """Bytes que ocupa una carpeta. Silenciosa: que no se pueda medir el disco
    no es motivo para que la lista de analisis deje de responder."""
    try:
        return sum(f.stat().st_size for f in ruta.rglob("*") if f.is_file())
    except Exception:
        return 0


@app.put("/api/jobs/{job_id}/avance")
async def guardar_avance(job_id: str, avance: dict):
    """Guarda el trabajo manual sobre un analisis, sobrescribiendo el anterior.

    Es lo que permite retomar sin archivos: el usuario aprieta «Guardar avance»
    y el analisis queda listo para reabrirse tal cual, desde cualquier
    navegador. El archivo que ademas se descarga sigue existiendo, pero pasa a
    ser lo que siempre debio ser —un respaldo, y la forma de tener guardadas
    VARIAS alternativas del mismo analisis— en vez del unico camino de vuelta.
    """
    with Session(engine) as session:
        job = session.get(Job, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Análisis no encontrado")

        proys = avance.get("proyecciones") or []
        # El resumen se calcula ACA y se guarda dentro del propio avance, para
        # que la lista pueda mostrar "169 aprobadas" sin cargar megas de puntos.
        avance["_meta"] = {
            "guardado_en": datetime.utcnow().isoformat(timespec="seconds"),
            # Aprobada Y no descartada, la misma definicion que el contador de
            # la vista: una aprobada que despues se descarto a mano conserva su
            # marca —descartar es reversible— pero ya no es parte de la cosecha.
            # Contandolas todas, el chip de la pantalla de entrada decia "169
            # aprobadas" para un avance donde quedaban 160.
            "aprobadas": sum(1 for t in proys
                             if t.get("aprobada") and t.get("estado") != "descartada"),
            "trayectorias": len(proys),
        }
        job.avance = avance
        session.add(job)
        session.commit()
        return {"ok": True, **avance["_meta"]}


@app.get("/api/jobs/{job_id}/avance")
def leer_avance(job_id: str):
    with Session(engine) as session:
        job = session.get(Job, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Análisis no encontrado")
        if not job.avance:
            raise HTTPException(status_code=404, detail="Este análisis no tiene avance guardado")
        return job.avance


@app.delete("/api/jobs/{job_id}")
def borrar_job(job_id: str):
    """Borra un analisis y sus archivos.

    Hasta ahora la unica forma de liberar espacio era la opcion «CERRAR Y BORRAR
    TODO» del .bat, que se lleva TAMBIEN los analisis que uno queria conservar.
    Poder borrar de a uno es la diferencia entre administrar el disco y perderlo
    todo para recuperar unos megas.
    """
    with Session(engine) as session:
        job = session.get(Job, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Análisis no encontrado")
        # is_running NO alcanza para saber si hay algo computando: un job pausado
        # esperando el percentil la tiene en True (para el WebSocket) y sin embargo
        # su hilo ya termino. Si eso contara como "corriendo", ese analisis seria
        # imborrable para siempre.
        if job.is_running and job.status != ESPERANDO_PERCENTIL:
            raise HTTPException(status_code=409, detail="El análisis todavía está corriendo")
        carpeta = ((job.entrada or {}).get("artefactos") or {}).get("carpeta")
        session.delete(job)
        session.commit()

    liberado = 0
    if carpeta:
        # El id va en la ruta, asi que se comprueba que la carpeta a borrar sea
        # EXACTAMENTE la del job y no algo que se le parezca: un rmtree guiado
        # por un parametro de la URL merece esa paranoia.
        destino = (Path(TEMP_VIDEOS) / carpeta).resolve()
        base = Path(TEMP_VIDEOS).resolve()
        if destino.parent == base and destino.name == carpeta and destino.is_dir():
            liberado = _peso_carpeta(destino)
            shutil.rmtree(destino, ignore_errors=True)
    return {"ok": True, "bytes_liberados": liberado}


@app.get("/api/results/{job_id}")
def get_job_results(job_id: str):
    with Session(engine) as session:
        # Buscamos el registro en la base de datos usando el UUID
        job = session.get(Job, job_id)
        
        if not job:
            # Si no existe, devolvemos un error 404 (Not Found)
            raise HTTPException(status_code=404, detail="Análisis no encontrado")
        
        # FastAPI automáticamente convierte el modelo Job de SQLModel a JSON
        return job

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)