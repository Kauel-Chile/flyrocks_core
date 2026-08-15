"""Prueba la purga de la cache por nodo sobre una cache sintetica.

    uv run python debug/probar_purga.py

Lo que de verdad hay que verificar no es que borre, sino que NO borre de mas:
los objetos de la cache estan DEDUPLICADOS y compartidos entre entradas (es lo
que baja una corrida de 659 MB a 60 MB), asi que al soltar una entrada vieja la
purga tiene que respetar todo lo que siga referenciado por una entrada viva.

El otro caso que cubre: un techo mas chico que una sola corrida no puede
llevarse lo recien calculado, o la cache queda escribiendo y borrando lo mismo
en cada vuelta sin dar nunca un HIT.
"""
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TMP = ROOT / "debug" / "out" / "_cache_prueba"
if TMP.exists():
    shutil.rmtree(TMP)
os.environ["DATA_DIR"] = str(TMP)

sys.path.insert(0, str(ROOT / "src"))

from utils.nodes import base  # noqa: E402

ok, mal = [], []
def afirmar(c, n):
    (ok if c else mal).append(n)

CACHE = base.CACHE_DIR
OBJ = base.OBJ_DIR
OBJ.mkdir(parents=True, exist_ok=True)

MB = 1024 * 1024

def objeto(nombre, mb):
    (OBJ / f"{nombre}.bin").write_bytes(b"x" * int(mb * MB))

def entrada(nombre, hashes, edad_s):
    p = CACHE / f"{nombre}.pkl"
    p.write_bytes(pickle.dumps({f"c{i}": h for i, h in enumerate(hashes)}))
    t = time.time() - edad_s
    os.utime(p, (t, t))
    return p

# --- escenario -------------------------------------------------------------
# 3 entradas. `compartido` lo usan la vieja y la nueva: la purga NO puede
# borrarlo al llevarse la vieja.
objeto("compartido", 10)
objeto("soloviejo", 10)
objeto("solomedio", 10)
objeto("solonuevo", 10)

vieja = entrada("vieja", ["compartido", "soloviejo"], edad_s=3000)
medio = entrada("medio", ["solomedio"],               edad_s=2000)
nueva = entrada("nueva", ["compartido", "solonuevo"], edad_s=10)

# --- 1. bajo el techo: no toca nada ----------------------------------------
liberado = base.purgar(limite_mb=100)
afirmar(liberado == 0, "bajo el techo no purga")
afirmar(len(list(OBJ.glob("*.bin"))) == 4, "bajo el techo conserva los 4 objetos")

# --- 2. techo que obliga a soltar entradas ---------------------------------
# Total = 40 MB, techo 25. Soltar solo `vieja` libera 10 MB (soloviejo) y deja
# 30: sigue pasada. Asi que tambien tiene que irse `medio`. Lo que NO puede
# pasar es que se lleve `compartido`, que la entrada nueva sigue usando.
liberado = base.purgar(limite_mb=25)
afirmar(liberado > 0, "sobre el techo purga algo")
afirmar(not vieja.exists(), "se fue la entrada mas vieja")
afirmar(not medio.exists(), "y tambien la del medio, porque con una no alcanzaba")
afirmar(nueva.exists(), "sobrevive la mas nueva")
afirmar(not (OBJ / "soloviejo.bin").exists(), "se borro el objeto que solo usaba la vieja")
afirmar(not (OBJ / "solomedio.bin").exists(), "y el que solo usaba la del medio")
afirmar((OBJ / "compartido.bin").exists(),
        "SE CONSERVA el objeto compartido con una entrada viva")
afirmar((OBJ / "solonuevo.bin").exists(), "se conserva el objeto de la nueva")

# --- 3. techo por debajo de una corrida: no se canibaliza -------------------
# Queda solo `nueva`, que pesa 20 MB. Con techo 11 no hay nada que se pueda
# soltar sin borrar lo recien calculado: se avisa y se deja pasada del techo.
liberado = base.purgar(limite_mb=11)
afirmar(nueva.exists(), "con el techo bajo la ultima entrada NO se purga")
afirmar((OBJ / "compartido.bin").exists() and (OBJ / "solonuevo.bin").exists(),
        "la ultima entrada conserva TODOS sus objetos")
afirmar(liberado == 0, "y no se libero nada, porque no habia nada purgable")

# --- 4. limite 0 = sin purga ------------------------------------------------
antes = len(list(OBJ.glob("*.bin")))
afirmar(base.purgar(limite_mb=0) == 0, "limite 0 no purga nunca")
afirmar(len(list(OBJ.glob("*.bin"))) == antes, "limite 0 no borra nada")

# --- 5. indice ilegible no rompe -------------------------------------------
(CACHE / "roto.pkl").write_bytes(b"no soy un pickle")
try:
    base.purgar(limite_mb=1)
    afirmar(True, "un indice corrupto no rompe la purga")
except Exception as e:
    afirmar(False, f"un indice corrupto rompio la purga: {e}")

print(f"\n  {len(ok)} bien, {len(mal)} mal\n")
for n in ok:
    print(f"    ok   {n}")
for n in mal:
    print(f"    MAL  {n}")

shutil.rmtree(TMP, ignore_errors=True)
sys.exit(1 if mal else 0)
