import os
import pandas as pd
import json

def load_supplementary_ais(planned_csv, actual_csv, output_json):
    """
    Loads Wakashio's verified intended passage plan and actual deviation tracks,
    and formats them for the Phase 4 scoring logic.
    """
    if not os.path.exists(planned_csv) or not os.path.exists(actual_csv):
        print("FAIL: Missing planned or actual track CSVs.")
        return False
        
    planned_df = pd.read_csv(planned_csv)
    actual_df = pd.read_csv(actual_csv)
    
    # We output a combined JSON representing the vessel's temporal/spatial ground truth
    # and its planned vs actual behavior.
    
    # Convert actual track to a list of dicts
    actual_track = []
    for _, row in actual_df.iterrows():
        actual_track.append({
            "time_lt": row["Time_LT"],
            "lat": row["Latitude"],
            "lon": row["Longitude"],
            "event": row["Event"]
        })
        
    # Convert planned to a list of dicts
    planned_track = []
    for _, row in planned_df.iterrows():
        planned_track.append({
            "pos_no": row["POS_No"],
            "lat": row["Latitude"],
            "lon": row["Longitude"],
            "course": row["Course"],
            "distance": row["Distance_NM"],
            "remarks": str(row["Remarks"]) if pd.notna(row["Remarks"]) else ""
        })
        
    # The actual final point (Grounding) is our anchor
    grounding = actual_track[-1]
    
    vessel_data = {
        "vessel_name": "Wakashio",
        "imo": "9337119",
        "grounding_anchor": grounding,
        "actual_deviation_track": actual_track,
        "planned_passage": planned_track
    }
    
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(vessel_data, f, indent=2)
        
    print(f"PASS: Loaded Wakashio verified ground-truth. Grounding anchor: {grounding['lat']}, {grounding['lon']}")
    return True

if __name__ == "__main__":
    load_supplementary_ais(
        "phase3_ais/planned_passage_waypoints.csv",
        "phase3_ais/actual_deviation_track.csv",
        "phase3_ais/wakashio_ground_truth.json"
    )
