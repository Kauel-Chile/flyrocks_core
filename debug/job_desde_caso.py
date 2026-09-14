"""Siembra un caso congelado como un JOB del core, para poder mostrarlo.

Es el inverso de `caso_desde_job.py`: aquel congela un analisis del core en
disco para poder iterar sin backend; este toma uno congelado y lo devuelve a la
base, con sus artefactos en su carpeta, como si el pipeline lo acabara de
producir.

Para que existe: una demo necesita que la lista de analisis del core tenga algo
que valga la pena abrir. Los jobs viejos de una base de desarrollo suelen ser
anteriores a la mitad de los campos que la vista usa hoy —sin homografia, sin
malla, sin artefactos por job— asi que abrirlos termina en un error o en una
pantalla a medias. Correr el pipeline de nuevo solo para tener con que mostrar
cuesta minutos de IA sobre un video de 4K.

Un caso congelado, en cambio, ya tiene todo lo que hace falta y no cuesta nada
volver a insertarlo: homografia, zonas, malla de tiros, mascara, frame y clip.

    uv run python debug/job_desde_caso.py ia-v9
    uv run python debug/job_desde_caso.py ia-v9 --nombre "Tronadura 3160-789"

Despues:  levantar el core y abrir el asistente; el analisis aparece en la
pantalla de entrada, listo para abrir sin reprocesar nada.
"""
import argparse
import json
import shutil
import sys
import uuid
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.utils.database import Job, engine, migrar, DATA_DIR  # noqa: E402
from sqlmodel import Session, SQLModel  # noqa: E402

CASOS = Path(__file__).resolve().parent / "casos"
TEMP = Path(DATA_DIR) / "temp_videos"


def log(msg):
    print(msg, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("caso", help="nombre de la carpeta en debug/casos/")
    ap.add_argument("--nombre", help="con que nombre aparece en la lista "
                                     "(por defecto, el del video del caso)")
    ap.add_argument("--job", help="forzar un id de job (por defecto uno nuevo)")
    args = ap.parse_args()

    carpeta_caso = CASOS / args.caso
    ruta = carpeta_caso / "caso.json"
    if not ruta.exists():
        raise SystemExit(f"no existe {ruta}")

    caso = json.loads(ruta.read_text(encoding="utf-8"))
    job_id = args.job or str(uuid.uuid4())
    destino = TEMP / job_id
    destino.mkdir(parents=True, exist_ok=True)

    # Los artefactos con los MISMOS nombres que usa el pipeline, porque la vista
    # los pide por esas rutas y no tiene por que enterarse de que este job no
    # nacio de una corrida.
    artefactos = {"carpeta": job_id}
    copias = [
        ("mascara.png", "mascara_cambios.png", "mascara"),
        ("frame.png", "frame.jpg", "frame"),
        # El clip que el navegador sabe reproducir. El `clip.mp4` crudo sale del
        # fourcc mp4v de OpenCV y ningun navegador lo pinta.
        ("clip_web.mp4", "clip_web.mp4", "video"),
    ]
    for origen, nombre, clave in copias:
        src = carpeta_caso / origen
        if not src.exists():
            log(f"  (falta {origen}: el job queda sin {clave})")
            artefactos[clave] = None
            continue
        shutil.copy2(src, destino / nombre)
        artefactos[clave] = f"{job_id}/{nombre}"
        log(f"  {origen} -> temp_videos/{job_id}/{nombre}  ({src.stat().st_size/1e6:.1f} MB)")

    malla = caso.get("malla") or {}
    meta = caso.get("meta") or {}
    calibra = caso.get("calibra") or {}
    zonas = caso.get("zonas") or {}

    ancla = meta.get("frame_detonacion")
    entrada = {
        "video": args.nombre or meta.get("video") or args.caso,
        "h_matrix": calibra.get("h_matrix"),
        "origin_zone": zonas.get("origen") or [],
        "expected_projection_zone": zonas.get("seguridad") or [],
        "parametros": (caso.get("pipeline") or {}).get("parametros")
                      or {"percentile": 99.0, "sigma": 0.5, "esp": 3.0},
        "artefactos": artefactos,
        "malla": {"meta": malla.get("meta") or {}, "pozos": malla.get("pozos") or []},
        # El ancla que la vista usa para cruzar el CSV con el clip. Se guardan
        # los tres numeros como en un analisis real, aunque aca los dos crudos
        # se derivan del ancla y no al reves.
        "recorte": {"frame_detonacion": ancla, "frame_inicio_corte": 0,
                    "ancla_frames": ancla} if ancla is not None else None,
        "sembrado_de": args.caso,
    }
    entrada = {k: v for k, v in entrada.items() if v is not None}

    # El core entrega las trayectorias indexadas por track_id; la vista las
    # normaliza a lista. Se rehace ese formato para que el camino de lectura sea
    # exactamente el mismo que el de un analisis de verdad.
    json_data = {}
    for t in caso.get("proyecciones") or []:
        if t.get("fuente") not in (None, "pipeline"):
            continue          # lo editado a mano no viene del pipeline
        json_data[str(t["id"])] = {
            "puntos": t.get("puntos") or [],
            "frames": t.get("frames"),
            "clasificacion": t.get("clasificacion"),
            "distancia_m": t.get("distancia_m"),
            "tortuosidad": t.get("tortuosidad"),
            "escape_relativo": t.get("escape_relativo"),
            "r2_score": t.get("r2_score"),
        }

    SQLModel.metadata.create_all(engine)
    migrar()
    with Session(engine) as s:
        if s.get(Job, job_id):
            raise SystemExit(f"el job {job_id} ya existe en la base")
        s.add(Job(id=job_id, creado_en=datetime.utcnow(), is_running=False,
                  status="Completado (sembrado desde un caso congelado)",
                  progress=100, result_file_path=None,
                  json_data=json_data, entrada=entrada))
        s.commit()

    log("")
    log(f"job {job_id}")
    log(f"  {len(json_data)} trayectorias · {len(entrada['malla']['pozos'])} pozos "
        f"· ancla {ancla} · nombre «{entrada['video']}»")
    log(f"  aparece en la pantalla de entrada del asistente, y directo en:")
    log(f"    /wizard/step-5?job={job_id}")


if __name__ == "__main__":
    main()
