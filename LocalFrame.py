from pyproj import Transformer

class LocalFrame:
    def __init__(self, ref_lat, ref_lon):
        # Create transformer: WGS84 → ENU at reference point
        self.transformer = Transformer.from_crs(
            "epsg:4326",  # WGS84
            f"+proj=tmerc +lat_0={ref_lat} +lon_0={ref_lon} +datum=WGS84",
            always_xy=True
        )

    def latlon_to_enu(self, lat, lon):
        x, y = self.transformer.transform(lon, lat)
        return x, y
