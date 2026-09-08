import os
import json
import argparse
from datetime import datetime

def convert_slick_to_seed(georef_path, output_path, timestamp_str, mask_source="predicted"):
    if not os.path.exists(georef_path):
        print(f"Error: {georef_path} not found.")
        return False
        
    with open(georef_path, 'r') as f:
        data = json.load(f)
        
    # Extract centroid
    centroid = None
    if "type" in data and data["type"] == "FeatureCollection":
        props = data["features"][0].get("properties", {})
        if "centroid_lon" in props and "centroid_lat" in props:
            centroid = [props["centroid_lon"], props["centroid_lat"]]
        elif "centroid" in props:
            centroid = props.get("centroid")
    elif "geo_centroid" in data:
        centroid = data["geo_centroid"]
        
    if not centroid:
        print("Error: Could not find centroid in georeferenced data.")
        return False
        
    lon, lat = centroid
    
    seed_config = {
        "lon": lon,
        "lat": lat,
        "time": timestamp_str,
        "uncertainty_radius": 1000, # 1km uncertainty for the centroid
        "mask_source": mask_source
    }
    
    if mask_source == "ground_truth":
        print("\n=========================================================================")
        print("                           [SYNTHETIC_TEST_DATA]                         ")
        print(" WARNING: USING GROUND-TRUTH MASK - NOT MODEL PREDICTION                 ")
        print(" THIS IS FOR PIPELINE VALIDATION ONLY. DO NOT PRESENT AS REAL RESULTS.   ")
        print("=========================================================================\n")
        seed_config["SYNTHETIC_TEST_DATA"] = True
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(seed_config, f, indent=2)
        
    print(f"Successfully converted slick to origin seed: Lon {lon}, Lat {lat} at {timestamp_str}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert georeferenced slick to OpenDrift seed")
    parser.add_argument("--georef", required=True, help="Path to georeferenced_slick.geojson")
    parser.add_argument("--output", required=True, help="Path to output opendrift_seed.json")
    parser.add_argument("--time", required=True, help="Time of slick detection (ISO 8601)")
    parser.add_argument("--mask-source", default="predicted", choices=["predicted", "ground_truth"], help="Source of the oil mask (predicted or ground_truth)")
    
    args = parser.parse_args()
    convert_slick_to_seed(args.georef, args.output, args.time, mask_source=args.mask_source)
