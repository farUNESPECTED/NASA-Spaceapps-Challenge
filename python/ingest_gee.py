import os
import ee
import pandas as pd


def init_gee(project_id: str):
    """Autentificare si initializare Google Earth Engine."""
    try:
        ee.Initialize(project=project_id)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project_id)


def mask_clouds_c2(image: ee.Image) -> ee.Image:
    """Elimina norii, umbrele si pixelii saturati (USGS Collection 2)."""
    qa = image.select('QA_PIXEL')
    clear_mask = (
        qa.bitwiseAnd(1 << 1).eq(0)
        .And(qa.bitwiseAnd(1 << 3).eq(0))
        .And(qa.bitwiseAnd(1 << 4).eq(0))
    )
    return image.updateMask(clear_mask)


def preprocess_l457(image: ee.Image) -> ee.Image:
    """Calibrare radiometrica si calcul indici pentru Landsat 4, 5 si 7 (1984 - 2012)."""
    img = mask_clouds_c2(image)

    green = img.select('SR_B2').multiply(0.0000275).add(-0.2)
    red = img.select('SR_B3').multiply(0.0000275).add(-0.2)
    nir = img.select('SR_B4').multiply(0.0000275).add(-0.2)
    swir1 = img.select('SR_B5').multiply(0.0000275).add(-0.2)

    lst_c = (
        img.select('ST_B6')
        .multiply(0.00341802)
        .add(149.0)
        .subtract(273.15)
        .rename('lst_c')
    )

    ndwi = green.subtract(nir).divide(green.add(nir)).rename('ndwi')
    ndvi = nir.subtract(red).divide(nir.add(red)).rename('ndvi')
    salinity_idx = swir1.subtract(nir).divide(swir1.add(nir)).rename('salinity_index')

    return ee.Image.cat([ndwi, ndvi, lst_c, salinity_idx]).copyProperties(image, ['system:time_start'])


def preprocess_l89(image: ee.Image) -> ee.Image:
    """Calibrare radiometrica si calcul indici pentru Landsat 8 si 9 (2013 - 2026)."""
    img = mask_clouds_c2(image)

    green = img.select('SR_B3').multiply(0.0000275).add(-0.2)
    red = img.select('SR_B4').multiply(0.0000275).add(-0.2)
    nir = img.select('SR_B5').multiply(0.0000275).add(-0.2)
    swir1 = img.select('SR_B6').multiply(0.0000275).add(-0.2)

    lst_c = (
        img.select('ST_B10')
        .multiply(0.00341802)
        .add(149.0)
        .subtract(273.15)
        .rename('lst_c')
    )

    ndwi = green.subtract(nir).divide(green.add(nir)).rename('ndwi')
    ndvi = nir.subtract(red).divide(nir.add(red)).rename('ndvi')
    salinity_idx = swir1.subtract(nir).divide(swir1.add(nir)).rename('salinity_index')

    return ee.Image.cat([ndwi, ndvi, lst_c, salinity_idx]).copyProperties(image, ['system:time_start'])


def build_harmonized_collection(aoi: ee.Geometry) -> ee.ImageCollection:
    """Imbina arhivele Landsat 4, 5, 7, 8 si 9 intr-un catalog unificat (1984 - 2026)."""
    l4 = ee.ImageCollection('LANDSAT/LT04/C02/T1_L2').filterBounds(aoi).filter(ee.Filter.lt('CLOUD_COVER', 35)).map(preprocess_l457)
    l5 = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2').filterBounds(aoi).filter(ee.Filter.lt('CLOUD_COVER', 35)).map(preprocess_l457)
    l7 = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2').filterBounds(aoi).filter(ee.Filter.lt('CLOUD_COVER', 35)).map(preprocess_l457)
    l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').filterBounds(aoi).filter(ee.Filter.lt('CLOUD_COVER', 35)).map(preprocess_l89)
    l9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2').filterBounds(aoi).filter(ee.Filter.lt('CLOUD_COVER', 35)).map(preprocess_l89)

    return l4.merge(l5).merge(l7).merge(l8).merge(l9)


def create_sub_tiles(bbox: list, nx: int = 2, ny: int = 2) -> list:
    """Imparte bazinul intr-o matrice de sub-regiuni pentru a ocoli limitele de memorie GEE."""
    min_lon, min_lat, max_lon, max_lat = bbox
    lon_step = (max_lon - min_lon) / nx
    lat_step = (max_lat - min_lat) / ny

    tiles = []
    for i in range(nx):
        for j in range(ny):
            t_min_lon = min_lon + i * lon_step
            t_max_lon = min_lon + (i + 1) * lon_step
            t_min_lat = min_lat + j * lat_step
            t_max_lat = min_lat + (j + 1) * lat_step
            tiles.append(ee.Geometry.Rectangle([t_min_lon, t_min_lat, t_max_lon, t_max_lat]))
    return tiles


def fetch_features_unbounded(fc: ee.FeatureCollection) -> list:
    """Descarca toate elementele dintr-un FeatureCollection folosind paginarea computeFeatures."""
    records = []
    page_token = None

    while True:
        params = {'expression': fc}
        if page_token:
            params['pageToken'] = page_token

        response = ee.data.computeFeatures(params)
        features = response.get('features', [])
        for f in features:
            records.append(f['properties'])

        page_token = response.get('nextPageToken')
        if not page_token:
            break

    return records


def fetch_full_aral_high_density(
    bbox: list,
    start_year: int = 1984,
    end_year: int = 2026,
    grid_scale_m: int = 4000,
    output_path: str = "raw_landsat_grid.parquet"
) -> pd.DataFrame:
    """Extrage întregul bazin Aral la densitate mare, cu salvare incrementala dupa fiecare an."""
    aoi_full = ee.Geometry.Rectangle(bbox)
    full_collection = build_harmonized_collection(aoi_full)
    tiles = create_sub_tiles(bbox, nx=2, ny=2)

    all_records = []

    for year_int in range(start_year, end_year + 1):
        start_date = ee.Date.fromYMD(year_int, 5, 15)
        end_date = ee.Date.fromYMD(year_int, 10, 15)

        yearly_col = full_collection.filterDate(start_date, end_date)
        count = yearly_col.size().getInfo()
        if count == 0:
            print(f" -> Anul {year_int}: 0 scene disponibile (sarit).")
            continue

        composite = yearly_col.median()
        timestamp_iso = f"{year_int}-08-01T00:00:00Z"
        year_records = []

        for idx, tile_geom in enumerate(tiles):
            sample_grid = tile_geom.coveringGrid(proj='EPSG:4326', scale=grid_scale_m)

            sampled = composite.select(['ndwi', 'ndvi', 'lst_c', 'salinity_index']).reduceRegions(
                collection=sample_grid,
                reducer=ee.Reducer.mean(),
                scale=150,
                tileScale=4
            )

            def attach_coords(feature):
                coords = feature.geometry().centroid(maxError=1).coordinates()
                return feature.set({
                    'timestamp': timestamp_iso,
                    'year': year_int,
                    'longitude': coords.get(0),
                    'latitude': coords.get(1)
                })

            valid_sampled = (
                sampled.filter(ee.Filter.notNull(['ndwi', 'lst_c']))
                .map(attach_coords)
                .select(['timestamp', 'year', 'longitude', 'latitude', 'ndwi', 'lst_c', 'ndvi', 'salinity_index'], retainGeometry=False)
            )

            tile_records = fetch_features_unbounded(valid_sampled)
            year_records.extend(tile_records)

        all_records.extend(year_records)
        print(f" -> Anul {year_int}: extras {len(year_records)} puncte (Total acumulat: {len(all_records)} randuri).")

        # Checkpoint incremental pe disc dupa fiecare an procesat
        if all_records:
            temp_df = pd.DataFrame(all_records)
            temp_df['latitude'] = temp_df['latitude'].round(4)
            temp_df['longitude'] = temp_df['longitude'].round(4)
            cols = ['timestamp', 'year', 'latitude', 'longitude', 'ndwi', 'lst_c', 'ndvi', 'salinity_index']
            temp_df = temp_df[cols].sort_values(['latitude', 'longitude', 'year']).reset_index(drop=True)
            temp_df.to_parquet(output_path, engine='pyarrow', index=False)

    return pd.DataFrame(all_records)


if __name__ == '__main__':
    init_gee('hazel-env-513413-n9')

    # Coordonate complete pentru intregul bazin istoric al Marii Aral 
    aral_full_bbox = [58.0, 43.4, 61.6, 47.0]
    output_file = "raw_landsat_grid.parquet"

    print("Pornire extractie masiva pentru TOT BAZINUL ARAL (1984 - 2026)...")
    print("Grila: 4 km (4 sectoare x ~2000 puncte/sector -> ~8000 puncte/an).")

    final_df = fetch_full_aral_high_density(
        bbox=aral_full_bbox,
        start_year=1984,
        end_year=2026,
        grid_scale_m=4000,
        output_path=output_file
    )

    if final_df.empty:
        raise RuntimeError("Eroare: Nu s-au returnat date.")

    print(f"\nExtractie completa : {output_file}")
    print(f"Total observatii generate: {len(final_df)}")
    print(f"Celule spatiale unice monitorizate: {final_df[['latitude', 'longitude']].drop_duplicates().shape[0]}")
    print("\nMostra din setul de date:")
    print(final_df.head(10))
