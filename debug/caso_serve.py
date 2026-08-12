"""
Sirve `debug/` por HTTP para que la vista pueda cargar un caso.

Hace falta porque abrir el HTML con doble clic (file://) hace que el navegador
bloquee por CORS cualquier fetch a los archivos del caso. La demo vieja se
libraba de esto porque tenia la malla embebida; con la mascara y el clip
adentro el HTML se iria a decenas de MB.

    uv run python debug/caso_serve.py
    -> http://localhost:8770/demo/vista.html
"""
import http.server
import socketserver
import sys
import threading
import webbrowser
from functools import partial
from pathlib import Path

DIR = Path(__file__).resolve().parent
PUERTO = 8770
INICIO = "demo/vista.html"


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Sin cache: si no, editas el caso o el HTML y el navegador te sigue
        # mostrando el anterior, que es media hora de confusion garantizada.
        self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()

    def log_message(self, fmt, *args):
        if "304" not in fmt % args:
            print(f"  {fmt % args}", flush=True)


class Servidor(socketserver.ThreadingTCPServer):
    """Un hilo por peticion.

    Con el TCPServer normal (una peticion a la vez) el navegador se queda con
    el servidor mientras baja el frame de 12 MB, y todo lo demas espera en cola
    hasta agotar su timeout: la vista parece colgada. Con imagenes de este
    tamaño el servidor concurrente no es un lujo.
    """
    # OJO: `allow_reuse_address` queda en False a proposito. En Windows esa
    # opcion permite que DOS procesos escuchen el mismo puerto a la vez, y las
    # conexiones caen en uno u otro de forma impredecible: media hora perdida
    # persiguiendo por que el navegador cargaba y curl daba timeout. Mejor que
    # el segundo arranque falle con un mensaje claro.
    daemon_threads = True


def ya_hay_uno():
    """True si algo ya escucha en el puerto."""
    import socket
    with socket.socket() as s:
        s.settimeout(0.4)
        return s.connect_ex(("127.0.0.1", PUERTO)) == 0


def main():
    abrir = "--no-abrir" not in sys.argv

    if ya_hay_uno():
        print(f"\n  Ya hay un servidor en el puerto {PUERTO}.")
        print(f"  Abre http://localhost:{PUERTO}/{INICIO}")
        print(f"  (si quieres reiniciarlo, cierra el otro con ctrl-c primero)\n")
        return
    with Servidor(("", PUERTO), partial(Handler, directory=str(DIR))) as s:
        url = f"http://localhost:{PUERTO}/{INICIO}"
        print(f"\n  sirviendo {DIR}\n  -> {url}\n  (ctrl-c para parar)\n")
        if abrir:
            # En un hilo aparte: si el navegador tarda en levantarse, el
            # servidor ya esta aceptando conexiones igual.
            threading.Thread(target=lambda: webbrowser.open(url), daemon=True).start()
        try:
            s.serve_forever()
        except KeyboardInterrupt:
            print("\n  chao\n")


if __name__ == "__main__":
    main()
