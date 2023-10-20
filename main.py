import geopandas as gpd

gdf = gpd.read_file('06/PRG_PunktyAdresowe_06.shp')
print(gdf.shape)
print(gdf.head())