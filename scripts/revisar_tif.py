# revisar_tif.py
# Muestra la informacion basica de un GeoTIFF para saber con que estamos trabajando.
# Uso:  python scripts/revisar_tif.py datos_crudos/dem_ctx.tif

import sys
import numpy as np
import rasterio

ruta = sys.argv[1]

with rasterio.open(ruta) as ds:
    print("Archivo:", ruta)
    print("Tamano (ancho x alto):", ds.width, "x", ds.height, "pixeles")
    print("Bandas:", ds.count, "| tipo:", ds.dtypes[0])
    print("Resolucion:", ds.res, "metros por pixel")
    print("Limites (en metros):", ds.bounds)
    print("Valor sin dato (nodata):", ds.nodata)
    print("Proyeccion:", ds.crs.to_string()[:200])

    # Leemos una version reducida para no llenar la memoria
    factor = max(1, ds.width // 1000)
    datos = ds.read(1, out_shape=(ds.height // factor, ds.width // factor), masked=True)
    datos = np.ma.masked_less(datos, -1e10)  # por si el nodata viene raro
    print("Valor minimo:", float(datos.min()), "| maximo:", float(datos.max()))
