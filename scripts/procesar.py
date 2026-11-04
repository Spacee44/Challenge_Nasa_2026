
# Recorta la zona de la caminata, calcula la pendiente y genera los archivos
# ligeros que usa la app web (PNG de cada capa + cuadricula JSON).
# Uso:  python scripts/procesar.py

import json
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform
from rasterio.windows import from_bounds
from matplotlib import colormaps
from PIL import Image

# ---------------- CONFIGURACION (lo que normalmente vas a mover) ----------------
DEM = "datos_crudos/dem_ctx.tif"       # modelo de elevacion (CTX, 20 m/pixel)
FOTO = "datos_crudos/foto_ctx.tif"     # ortofoto (CTX, 6 m/pixel)
SALIDA = Path("docs/data")             # aqui quedan los archivos para la web

CENTRO_LAT = 18.46     # centro de la zona de la caminata (grados norte)
CENTRO_LON = 77.42     # (grados este)
TAMANO_KM = 6          # lado del cuadrado que recortamos
PENDIENTE_MAX = 25     # grados; arriba de esto la celda queda bloqueada
LADO_FOTO = 1500       # pixeles maximos de la foto (para que pese poco)
RADIO_MARTE = 3396190  # metros, el mismo que usa USGS en estos archivos
# ---------------------------------------------------------------------------------

MARTE_LONLAT = f"+proj=longlat +R={RADIO_MARTE} +no_defs"


def a_metros(ds, lon, lat):
    """Convierte lon/lat (grados) a las coordenadas en metros del archivo."""
    xs, ys = transform(MARTE_LONLAT, ds.crs, [lon], [lat])
    return xs[0], ys[0]


def a_lonlat(ds, x, y):
    """Lo contrario: de metros del archivo a lon/lat."""
    lons, lats = transform(ds.crs, MARTE_LONLAT, [x], [y])
    return lons[0], lats[0]


def recortar(ruta, limites, lado_max=None):
    """Lee solo el cuadro que nos interesa. Si lado_max, lo reduce a ese tamano."""
    with rasterio.open(ruta) as ds:
        ventana = from_bounds(*limites, transform=ds.transform)
        alto, ancho = round(ventana.height), round(ventana.width)
        if lado_max and max(alto, ancho) > lado_max:
            f = lado_max / max(alto, ancho)
            alto, ancho = round(alto * f), round(ancho * f)
        datos = ds.read(1, window=ventana, out_shape=(alto, ancho),
                        masked=True, boundless=True).astype("float32")
        datos = np.ma.masked_less(datos, -1e10)  # nodata raro de USGS
        return datos.filled(np.nan)


def guardar_png(rgba, nombre):
    Image.fromarray(rgba).save(SALIDA / nombre)
    print("  guardado", SALIDA / nombre, rgba.shape[1], "x", rgba.shape[0])


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    mitad = TAMANO_KM * 1000 / 2

    with rasterio.open(DEM) as ds:
        cx, cy = a_metros(ds, CENTRO_LON, CENTRO_LAT)
        res_x, res_y = ds.res

        # En esta proyeccion un metro "del mapa" en x no es un metro real en el suelo.
        # Medimos cuanto se estira comparando 0.01 grados de longitud.
        x2, _ = a_metros(ds, CENTRO_LON + 0.01, CENTRO_LAT)
        metros_reales = RADIO_MARTE * math.cos(math.radians(CENTRO_LAT)) * math.radians(0.01)
        escala_x = metros_reales / (x2 - cx)

        # oeste, sur, este, norte (en x ensanchamos para que el cuadro sea parejo en el suelo)
        mitad_x = mitad / escala_x
        limites = (cx - mitad_x, cy - mitad, cx + mitad_x, cy + mitad)

        oeste, sur = a_lonlat(ds, limites[0], limites[1])
        este, norte = a_lonlat(ds, limites[2], limites[3])

    print("1) Recortando elevacion...")
    z = recortar(DEM, limites)
    alto, ancho = z.shape
    dx = res_x * escala_x  # metros reales entre columnas
    dy = res_y             # metros reales entre filas
    print(f"  {ancho} x {alto} celdas, celda de {dx:.1f} m x {dy:.1f} m")

    print("2) Calculando pendiente...")
    dz_dy, dz_dx = np.gradient(z, dy, dx)
    pendiente = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))
    sin_dato = np.isnan(pendiente)
    bloqueadas = (pendiente > PENDIENTE_MAX) | sin_dato
    print(f"  pendiente maxima: {np.nanmax(pendiente):.1f} grados")
    print(f"  celdas bloqueadas: {100 * bloqueadas.mean():.1f} %")

    print("3) Generando capas PNG...")
    # Elevacion con colores (transparente donde no hay dato)
    zmin, zmax = np.nanpercentile(z, [2, 98])
    norm = np.clip((z - zmin) / (zmax - zmin), 0, 1)
    rgba = (colormaps["terrain"](np.nan_to_num(norm)) * 255).astype("uint8")
    rgba[np.isnan(z), 3] = 0
    guardar_png(rgba, "elevacion.png")

    # Pendiente: de morado (plano) a amarillo (inclinado); rojo = bloqueado
    norm = np.clip(np.nan_to_num(pendiente) / PENDIENTE_MAX, 0, 1)
    rgba = (colormaps["viridis"](norm) * 255).astype("uint8")
    rgba[bloqueadas] = [220, 30, 30, 255]
    rgba[sin_dato, 3] = 0
    guardar_png(rgba, "pendiente.png")

    # Foto: estiramos el contraste para que se vea bien
    foto = recortar(FOTO, limites, lado_max=LADO_FOTO)
    p2, p98 = np.nanpercentile(foto, [2, 98])
    gris = (np.clip((foto - p2) / (p98 - p2), 0, 1) * 255)
    gris = np.nan_to_num(gris).astype("uint8")
    rgba = np.dstack([gris, gris, gris, np.where(np.isnan(foto), 0, 255).astype("uint8")])
    guardar_png(rgba, "foto.png")

    print("4) Guardando cuadricula para la ruta...")
    cuadricula = {
        "ancho": ancho,
        "alto": alto,
        "celda_x": round(dx, 3),
        "celda_y": round(dy, 3),
        # pendiente en decimas de grado (-1 = sin dato) para que el JSON pese menos
        "pendiente": np.where(sin_dato, -1, np.round(pendiente * 10)).astype(int).ravel().tolist(),
        "elevacion": np.where(np.isnan(z), -99999, np.round(z)).astype(int).ravel().tolist(),
    }
    (SALIDA / "cuadricula.json").write_text(json.dumps(cuadricula, separators=(",", ":")))

    info = {
        "ancho_m": round(ancho * dx, 1),
        "alto_m": round(alto * dy, 1),
        "oeste": oeste, "este": este, "sur": sur, "norte": norte,
        "pendiente_max": PENDIENTE_MAX,
    }
    (SALIDA / "info.json").write_text(json.dumps(info, indent=2))
    print("  listo:", SALIDA / "cuadricula.json", "y", SALIDA / "info.json")


if __name__ == "__main__":
    main()
