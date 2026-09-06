import os
import json
import matplotlib.pyplot as plt

def render_map():
    print("--- Phase 5: Rendering Visual Map ---")
    
    origin_path = "phase2_drift/outputs/origin.json"
    ground_truth_path = "phase3_ais/wakashio_ground_truth.json"
    
    if not os.path.exists(origin_path) or not os.path.exists(ground_truth_path):
        print("FAIL: Missing data files to render map.")
        return False
        
    with open(origin_path, 'r') as f:
        origin_data = json.load(f)
        
    with open(ground_truth_path, 'r') as f:
        vessel_data = json.load(f)
        
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 1. Plot the Planned Passage
    planned_lons = []
    planned_lats = []
    from evidence_score import parse_dm_to_dd
    
    for wp in vessel_data["planned_passage"]:
        if wp["pos_no"] in [22, 23, 24]: # Just the ones around Mauritius
            planned_lats.append(parse_dm_to_dd(wp["lat"]))
            planned_lons.append(parse_dm_to_dd(wp["lon"]))
            
    if planned_lons:
        ax.plot(planned_lons, planned_lats, 'k--', label="Planned Passage (Pilot to Pilot)", linewidth=2)
        
    # 2. Plot Actual Deviation Track
    actual_lons = []
    actual_lats = []
    for pt in vessel_data["actual_deviation_track"]:
        actual_lats.append(parse_dm_to_dd(pt["lat"]))
        actual_lons.append(parse_dm_to_dd(pt["lon"]))
        
    ax.plot(actual_lons, actual_lats, 'r-', label="Actual Deviation Track (July 25)", linewidth=2, marker='o')
    
    # 3. Grounding Point
    grounding = vessel_data["grounding_anchor"]
    g_lat = parse_dm_to_dd(grounding["lat"])
    g_lon = parse_dm_to_dd(grounding["lon"])
    ax.plot(g_lon, g_lat, 'rX', markersize=12, label="Actual Grounding Point (Reported)")
    
    # 4. OpenDrift Origin (Slick backtracking)
    drift_lon = origin_data["origin_lon"]
    drift_lat = origin_data["origin_lat"]
    ax.plot(drift_lon, drift_lat, 'b*', markersize=14, label="Predicted Spill Origin (Hindcast)")
    
    # 5. Drift Path (Mocked as straight line from simulated slick back to origin)
    # The slick seed was at 57.65, -20.35
    ax.plot([57.65, drift_lon], [-20.35, drift_lat], 'b:', label="Hindcast Backtrack Path")
    
    title = "Wakashio: Predicted Origin vs Official Trajectory Ground-Truth"
    if origin_data.get("SYNTHETIC_TEST_DATA"):
        title = "[SYNTHETIC_TEST_DATA] " + title
        # Add a big red text warning on the map
        ax.text(0.5, 0.95, "WARNING: SYNTHETIC DATA - NOT A PHYSICAL HINDCAST", 
                color='red', fontsize=14, fontweight='bold', ha='center', va='center', transform=ax.transAxes)
                
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc='upper right')
    ax.grid(True)
    
    os.makedirs("phase5_output", exist_ok=True)
    out_file = "phase5_output/wakashio_validation_map.png"
    plt.savefig(out_file, dpi=300)
    print(f"PASS: Rendered validation map to {out_file}")
    
    return True

if __name__ == "__main__":
    # Ensure evidence_score is importable for the parser helper
    import sys
    sys.path.append("phase4_matching")
    render_map()
