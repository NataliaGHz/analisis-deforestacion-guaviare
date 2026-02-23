from pathlib import Path
import rasterio
import pandas as pd
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape
import numpy as np


# ============================================
# REPROYECTAR RASTER
# ============================================

def reproyectar_raster(
    raster_entrada,
    raster_salida,
    crs_destino="EPSG:9377",
    metodo_resampling=Resampling.nearest
):
    """
    Reproyecta un raster a un nuevo sistema de referencia espacial.

    Parámetros
    ----------
    raster_entrada : str o Path
        Ruta del raster original.
    raster_salida : str o Path
        Ruta donde se guardará el raster reproyectado.
    crs_destino : str
        CRS destino (por defecto EPSG:9377 - MAGNA-SIRGAS Colombia).
    metodo_resampling : rasterio.warp.Resampling
        Método de remuestreo (nearest recomendado para datos categóricos).

    Retorna
    -------
    None
    """

    raster_entrada = Path(raster_entrada)
    raster_salida = Path(raster_salida)

    with rasterio.open(raster_entrada) as src:
        transform, width, height = calculate_default_transform(
            src.crs,
            crs_destino,
            src.width,
            src.height,
            *src.bounds
        )

        kwargs = src.meta.copy()
        kwargs.update({
            "crs": crs_destino,
            "transform": transform,
            "width": width,
            "height": height
        })

        raster_salida.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(raster_salida, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=crs_destino,
                    resampling=metodo_resampling
                )




# ============================================
# REPROYECTAR VECTOR
# ============================================

def reproyectar_vector(
    vector_entrada,
    vector_salida,
    crs_destino="EPSG:9377"
):
    """
    Reproyecta un archivo vectorial a un nuevo sistema de referencia espacial.

    Parámetros
    ----------
    vector_entrada : str o Path
        Ruta al archivo vectorial de entrada (shp, gpkg, geojson, etc.)
    vector_salida : str o Path
        Ruta del archivo vectorial reproyectado
    crs_destino : str, opcional
        CRS de destino en formato EPSG (por defecto EPSG:9377)

    Retorna
    -------
    None
    """

    vector_entrada = Path(vector_entrada)
    vector_salida = Path(vector_salida)

    # Leer vector
    gdf = gpd.read_file(vector_entrada)

    if gdf.crs is None:
        raise ValueError(f"El vector {vector_entrada.name} no tiene CRS definido")

    # Reproyectar solo si es necesario
    if gdf.crs.to_string() != crs_destino:
        gdf = gdf.to_crs(crs_destino)

    # Crear carpeta de salida
    vector_salida.parent.mkdir(parents=True, exist_ok=True)

    # Guardar (driver explícito)
    gdf.to_file(vector_salida, driver="GPKG")




# ============================================
# VECTORIZAR RASTER
# ============================================

def raster_a_vector(
    raster_path,
    vector_salida,
    campo_valor="clase",
    anio=None,
    campo_anio="anio",
    nodata=None
):
    """
    Vectoriza un raster categórico excluyendo valores NoData.

    Parámetros
    ----------
    raster_path : str o Path
        Ruta del raster de entrada.
    vector_salida : str o Path
        Ruta del archivo vectorial de salida (gpkg o shp).
    campo_valor : str
        Nombre del campo donde se guardará el valor del píxel.
    anio : int o str, opcional
        Año asociado al raster.
    campo_anio : str
        Nombre del campo del año.
    nodata : int o float, opcional
        Valor NoData. Si es None, se toma del raster.

    Retorna
    -------
    geopandas.GeoDataFrame
    """

    raster_path = Path(raster_path)
    vector_salida = Path(vector_salida)

    with rasterio.open(raster_path) as src:
        raster = src.read(1)
        transform = src.transform
        crs = src.crs

        if nodata is None:
            nodata = src.nodata

        # máscara para excluir NoData
        if nodata is not None:
            mask = raster != nodata
        else:
            mask = ~np.isnan(raster)

        geometries = []
        values = []

        for geom, value in shapes(raster, mask=mask, transform=transform):
            geometries.append(shape(geom))
            values.append(value)

    gdf = gpd.GeoDataFrame(
        {campo_valor: values},
        geometry=geometries,
        crs=crs
    )

    if anio is not None:
        gdf[campo_anio] = anio

    vector_salida.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(vector_salida, driver="GPKG")

    return gdf



# ============================================
# SUBDIVIDIR POR MUNICIPIO
# ============================================

def subdividir_por_municipios(
    gdf,
    municipios,
    campo_municipio
):
    """
    Intersecta polígonos de agua con municipios.

    Parámetros
    ----------
    gdf : GeoDataFrame
        Polígonos de agua.
    municipios : GeoDataFrame
        Capa de municipios.
    campo_municipio : str
        Campo con el nombre o código del municipio.

    Retorna
    -------
    GeoDataFrame
        Agua subdividida por municipio.
    """

    # Reproyección si es necesario
    if gdf.crs != municipios.crs:
        gdf = gdf.to_crs(municipios.crs)

    gdf = gpd.overlay(
        gdf,
        municipios[[campo_municipio, "geometry"]],
        how="intersection"
    )

    return gdf




# ============================================
# ESTANDARIZAR ATRIBUTOS
# ============================================

import geopandas as gpd

def estandarizar_atributos(
    gdf,
    reglas,
    iniciar_id=1
):
    """
    Estandariza atributos de una capa según reglas definidas.
    """

    gdf = gdf.copy()

    for campo_nuevo, regla in reglas.items():

        if regla.get("tipo") == "id":
            gdf[campo_nuevo] = range(iniciar_id, iniciar_id + len(gdf))

        elif "diccionario" in regla and "desde" in regla:
            gdf[campo_nuevo] = gdf[regla["desde"]].map(regla["diccionario"])

        elif "desde" in regla:
            gdf[campo_nuevo] = (
                gdf[regla["desde"]] if regla["desde"] in gdf.columns else None
            )

        elif "valor_fijo" in regla:
            gdf[campo_nuevo] = regla["valor_fijo"]

        else:
            gdf[campo_nuevo] = None

    return gdf




# ============================================
# UNIR CAPAS
# ============================================

def unir_capas_vectoriales(
    capas,
    crs_objetivo=None,
    campo_id="ID",
    iniciar_id=1
):
    """
    Une múltiples capas vectoriales con los mismos atributos
    y genera un ID único global.

    Parámetros
    ----------
    capas : list
        Lista de rutas (Path/str) o GeoDataFrames.
    crs_objetivo : str, opcional
        CRS destino (ej. "EPSG:9377").
    campo_id : str
        Nombre del campo ID a crear.
    iniciar_id : int
        Valor inicial del ID.

    Retorna
    -------
    GeoDataFrame
    """

    lista_gdf = []

    for capa in capas:
        if isinstance(capa, (str, Path)):
            gdf = gpd.read_file(capa)
        else:
            gdf = capa.copy()

        lista_gdf.append(gdf)

    if not lista_gdf:
        raise ValueError("La lista de capas está vacía")

    crs_base = crs_objetivo if crs_objetivo else lista_gdf[0].crs

    lista_gdf = [
        gdf.to_crs(crs_base) if gdf.crs != crs_base else gdf
        for gdf in lista_gdf
    ]

    gdf_unido = gpd.GeoDataFrame(
        pd.concat(lista_gdf, ignore_index=True),
        crs=crs_base
    )

    # 🔑 ID ÚNICO GLOBAL
    gdf_unido[campo_id] = range(iniciar_id, iniciar_id + len(gdf_unido))

    return gdf_unido



# ============================================
# CLIP VECTOR
# ============================================

def clip_a_area_estudio(
    capa,
    area_estudio,
    guardar=False,
    ruta_salida=None
):
    """
    Realiza un clip espacial de una capa vectorial a un área de estudio.
    """

    # Leer capas si vienen como rutas
    gdf = gpd.read_file(capa) if isinstance(capa, (str, Path)) else capa.copy()
    aoi = gpd.read_file(area_estudio) if isinstance(area_estudio, (str, Path)) else area_estudio.copy()

    # Verificar CRS
    if gdf.crs != aoi.crs:
        gdf = gdf.to_crs(aoi.crs)

    # Clip espacial
    gdf_clip = gpd.clip(gdf, aoi)

    # Guardar si se solicita
    if guardar:
        if ruta_salida is None:
            raise ValueError("Debes especificar 'ruta_salida' si guardar=True")
        gdf_clip.to_file(ruta_salida, driver="GPKG")

    return gdf_clip



# ============================================
# EXTRAER, VECTORIZAR Y DISOLVER UN VALOR DE PIXEL
# ============================================
def vectorizar_valor_pixel_por_anio(
    raster_path,
    pixel_value,
    year,
    nodata=None
):
    import rasterio
    import numpy as np
    import geopandas as gpd
    from rasterio.features import shapes
    from shapely.geometry import shape

    with rasterio.open(raster_path) as src:
        raster = src.read(1)
        transform = src.transform
        crs = src.crs
        if nodata is None:
            nodata = src.nodata

    mask = raster == pixel_value

    geometries = []
    for geom, val in shapes(
        raster.astype(np.int32),
        mask=mask,
        transform=transform
    ):
        geometries.append(shape(geom))

    if not geometries:
        return gpd.GeoDataFrame(
            columns=["year", "pixel_value", "geometry"],
            crs=crs
        )

    gdf = gpd.GeoDataFrame(
        {
            "year": [year] * len(geometries),
            "pixel_value": [pixel_value] * len(geometries)
        },
        geometry=geometries,
        crs=crs
    )

    # 🔥 Dissolve: une los polígonos que se tocan (por año)
    gdf = gdf.dissolve(by="year")
    gdf = gdf.reset_index()

    return gdf


