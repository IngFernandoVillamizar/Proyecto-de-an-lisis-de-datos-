"""Carga los datos en bruto de idealista18 (solo Valencia) en una base SQLite.

Una tabla por fichero de origen, sin limpiar nada: los valores se guardan tal
cual vienen. Las geometrias de R (sf) se guardan como texto WKT.

Requiere: rdata, pandas, pyarrow.
"""
import sqlite3
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rdata

warnings.filterwarnings("ignore")

RAIZ = Path(__file__).resolve().parents[1]
RAW = RAIZ / "data" / "raw" / "idealista18"
PARQUET = RAIZ / "data" / "interim" / "idealista18_venta.parquet"
BD = RAIZ / "data" / "valencia.sqlite"


def leer_rda(nombre):
    parsed = rdata.parser.parse_file(RAW / nombre)
    return rdata.conversion.convert(parsed, default_encoding="utf8")


def anillo(m):
    return "(" + ", ".join(f"{x!r} {y!r}" for x, y in np.asarray(m, dtype=float).tolist()) + ")"


def a_wkt(g):
    """Convierte una geometria sf (punto, poligono o multipoligono) a WKT."""
    if isinstance(g, np.ndarray) and g.ndim == 1:
        return f"POINT ({float(g[0])!r} {float(g[1])!r})"
    if isinstance(g[0], np.ndarray):
        return "POLYGON (" + ", ".join(anillo(r) for r in g) + ")"
    return "MULTIPOLYGON (" + ", ".join("(" + ", ".join(anillo(r) for r in p) + ")" for p in g) + ")"


def preparar(df):
    df = df.copy()
    if "geometry" in df.columns:
        df["geometry"] = df["geometry"].map(a_wkt)
    for c in df.columns:
        if str(df[c].dtype) in ("category", "string", "str"):
            df[c] = df[c].astype(object)
    return df.reset_index(drop=True)


def main():
    venta = leer_rda("Valencia_Sale.rda")["Valencia_Sale"]
    poligonos = leer_rda("Valencia_Polygons.rda")["Valencia_Polygons"]

    # Valencia_POIS es una lista de tres tablas Lon/Lat: se apilan con su tipo.
    pois = pd.concat(
        [t.assign(POI_TYPE=str(k)) for k, t in leer_rda("Valencia_POIS.rda")["Valencia_POIS"].items()],
        ignore_index=True,
    )[["POI_TYPE", "Lon", "Lat"]]

    distritos = leer_rda("properties_by_district.rda")["properties_by_district"]
    distritos = distritos[distritos["CITY"] == "Valencia"]

    parquet = pd.read_parquet(PARQUET)
    parquet = parquet[parquet["CITY"] == "Valencia"]

    tablas = {
        "valencia_sale": venta,
        "valencia_polygons": poligonos,
        "valencia_pois": pois,
        "properties_by_district": distritos,
        "idealista18_venta": parquet,
    }

    BD.unlink(missing_ok=True)
    with sqlite3.connect(BD) as con:
        for nombre, df in tablas.items():
            preparar(df).to_sql(nombre, con, index=False)
        for nombre, df in tablas.items():
            n = con.execute(f"SELECT COUNT(*) FROM {nombre}").fetchone()[0]
            print(f"{nombre}: {n} filas (origen {len(df)}), {len(df.columns)} columnas")
    con.close()


if __name__ == "__main__":
    main()
