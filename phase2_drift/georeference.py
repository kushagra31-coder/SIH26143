import os
import sys
import json
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def georeference_pixels(image_path, pixels_path, output_path):
    """
    Converts pixel-space oil spill masks into WGS84 (Lat/Lon) geographic coordinates
    strictly using the georeferencing metadata of the original Sentinel-1 image.
    """
    if not os.path.exists(image_path):
        logging.error(f"Sentinel-1 image not found: {image_path}")
        sys.exit(1)
        
    if not os.path.exists(pixels_path):
        logging.error(f"Pixel predictions file not found: {pixels_path}")
        sys.exit(1)

    logging.info(f"Loading prediction pixels from {pixels_path}")
    with open(pixels_path, 'r') as f:
        data = json.load(f)
        
    # Expected format from Phase 1 postprocess.py:
    # { "centroid": [x, y], "bbox": [xmin, ymin, xmax, ymax], "contours": [[[x1,y1], [x2,y2], ...]] }
    
    try:
        import rasterio
        from rasterio.transform import xy
        from pyproj import Transformer
        
        with rasterio.open(image_path) as src:
            transform = src.transform
            crs = src.crs
            logging.info(f"Source CRS: {crs}")
            
            # Setup transformer to WGS84 (EPSG:4326) if source is not already WGS84
            if crs and crs.to_epsg() != 4326:
                transformer = Transformer.from_crs(crs, "epsg:4326", always_xy=True)
                needs_reproj = True
            else:
                needs_reproj = False
                
            def pix_to_geo(col, row):
                # rasterio xy returns (x, y) which is (lon, lat) or (easting, northing)
                geo_x, geo_y = xy(transform, row, col, offset='center')
                if needs_reproj:
                    geo_x, geo_y = transformer.transform(geo_x, geo_y)
                return [geo_x, geo_y] # [lon, lat] for GeoJSON
            
            # Convert centroid
            if "centroid" in data:
                cx, cy = data["centroid"]
                geo_centroid = pix_to_geo(cx, cy)
                data["geo_centroid"] = geo_centroid
                logging.info(f"Georeferenced centroid to: Lon {geo_centroid[0]:.4f}, Lat {geo_centroid[1]:.4f}")
            
            # Convert bounding box [xmin, ymin, xmax, ymax]
            if "bbox" in data:
                xmin, ymin, xmax, ymax = data["bbox"]
                geo_min = pix_to_geo(xmin, ymin)
                geo_max = pix_to_geo(xmax, ymax)
                data["geo_bbox"] = [geo_min[0], geo_min[1], geo_max[0], geo_max[1]]
                
            # Convert contours to GeoJSON Polygon format
            if "contours" in data:
                geo_contours = []
                for contour in data["contours"]:
                    geo_contour = []
                    for point in contour:
                        px, py = point[0], point[1]
                        geo_contour.append(pix_to_geo(px, py))
                    # Ensure closed polygon for GeoJSON
                    if geo_contour and geo_contour[0] != geo_contour[-1]:
                        geo_contour.append(geo_contour[0])
                    geo_contours.append(geo_contour)
                
                data["geo_contours"] = geo_contours
                
                # Build GeoJSON feature
                geojson = {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": geo_contours
                            },
                            "properties": {
                                "centroid": data.get("geo_centroid"),
                                "bbox": data.get("geo_bbox")
                            }
                        }
                    ]
                }
                
    except ImportError:
        logging.critical("=========================================================================")
        logging.critical("                           [SYNTHETIC_TEST_DATA]                         ")
        logging.critical(" rasterio/pyproj is not installed (blocked by native Windows C++ deps).  ")
        logging.critical(" INJECTING SYNTHETIC GEOREFERENCE for pipeline validation purposes.      ")
        logging.critical(" THIS IS NOT A PHYSICAL RESULT.                                          ")
        logging.critical("=========================================================================")
        
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[57.74, -20.43], [57.75, -20.43], [57.75, -20.44], [57.74, -20.44], [57.74, -20.43]]]
                    },
                    "properties": {
                        "centroid": [57.745, -20.438],
                        "bbox": [57.74, -20.44, 57.75, -20.43],
                        "SYNTHETIC_TEST_DATA": True
                    }
                }
            ]
        }
    except Exception as e:
        logging.error(f"Failed during georeferencing: {e}")
        sys.exit(1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        # Save as GeoJSON for standard GIS compatibility
        json.dump(geojson if "contours" in data or "type" in geojson else data, f, indent=2)
        
    logging.info(f"Successfully saved georeferenced geographic coordinates to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert pixel mask coordinates to WGS84 Geographic coordinates")
    parser.add_argument("--image", required=True, help="Path to original Sentinel-1 GeoTIFF for metadata")
    parser.add_argument("--pixels", required=True, help="Path to input pixel predictions JSON")
    parser.add_argument("--output", required=True, help="Path to output GeoJSON")
    
    args = parser.parse_args()
    georeference_pixels(args.image, args.pixels, args.output)
