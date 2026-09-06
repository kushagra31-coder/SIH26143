import math
from datetime import datetime

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0 # Earth radius in kilometers
    
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c

def parse_dm_to_dd(dm_str):
    """
    Parses a string like "20°08.3'S" into decimal degrees.
    """
    if not isinstance(dm_str, str):
        return 0.0
    try:
        parts = dm_str.replace('°', ' ').replace("'", ' ').replace('S', ' S').replace('N', ' N').replace('E', ' E').replace('W', ' W').split()
        deg = float(parts[0])
        min = float(parts[1])
        dir = parts[2]
        
        dd = deg + min/60.0
        if dir in ['S', 'W']:
            dd *= -1
        return dd
    except Exception:
        return 0.0

def compute_spatial_score(origin_lat, origin_lon, vessel_lat, vessel_lon):
    """
    Calculates a score (0-100) based on how close the vessel's actual location 
    was to the backtracked spill origin.
    """
    dist_km = haversine_distance(origin_lat, origin_lon, vessel_lat, vessel_lon)
    # If distance is 0, score is 100. If distance > 50km, score drops towards 0.
    score = max(0, 100 - (dist_km * 2))
    return score, dist_km

def compute_behavior_deviation_score(planned_track, actual_track):
    """
    Calculates the Behaviour Evidence score.
    NOTE: As per the honest MVP scope, this explicitly substitutes the original "AIS-gap" concept.
    We are scoring the vessel's behaviour based on the magnitude of deviation from its official 
    Passage Plan (Pilot-to-Pilot) vs its actual recorded track.
    """
    # For MVP, we know the planned waypoint 23 (South Mauritius) vs actual grounding
    # We will find the grounding point in actual_track and the corresponding planned point
    
    actual_grounding = actual_track[-1]
    actual_lat = parse_dm_to_dd(actual_grounding['lat'])
    actual_lon = parse_dm_to_dd(actual_grounding['lon'])
    
    # Find planned waypoint nearest to Mauritius (pos 23 from our CSV)
    planned_wp = None
    for wp in planned_track:
        if wp.get('pos_no') == 23:
            planned_wp = wp
            break
            
    if not planned_wp:
        return 0, 0
        
    plan_lat = parse_dm_to_dd(planned_wp['lat'])
    plan_lon = parse_dm_to_dd(planned_wp['lon'])
    
    deviation_km = haversine_distance(plan_lat, plan_lon, actual_lat, actual_lon)
    
    # Anomalous behavior signal: deviation > 10km is highly suspicious for a bulk carrier
    score = min(100, deviation_km * 2) # Higher deviation = higher suspicion score (0-100)
    
    return score, deviation_km

def evaluate_vessel(origin_data, vessel_data):
    """
    Evaluates a single vessel against the origin data.
    """
    origin_lon = origin_data["origin_lon"]
    origin_lat = origin_data["origin_lat"]
    
    actual_track = vessel_data["actual_deviation_track"]
    planned_track = vessel_data["planned_passage"]
    
    # Get vessel position at grounding time (the anchor)
    anchor = vessel_data["grounding_anchor"]
    vessel_lat = parse_dm_to_dd(anchor['lat'])
    vessel_lon = parse_dm_to_dd(anchor['lon'])
    
    spatial_score, dist_km = compute_spatial_score(origin_lat, origin_lon, vessel_lat, vessel_lon)
    
    # Behavior Score based on Route Deviation
    behavior_score, deviation_km = compute_behavior_deviation_score(planned_track, actual_track)
    
    # Total Evidence Score (Weighted)
    total_score = (spatial_score * 0.6) + (behavior_score * 0.4)
    
    result = {
        "vessel_name": vessel_data["vessel_name"],
        "imo": vessel_data["imo"],
        "total_score": round(total_score, 2),
        "spatial_evidence": {
            "score": round(spatial_score, 2),
            "distance_to_origin_km": round(dist_km, 2),
            "note": "Distance from predicted drift origin to actual vessel location."
        },
        "behavior_evidence": {
            "score": round(behavior_score, 2),
            "deviation_from_plan_km": round(deviation_km, 2),
            "note": "EXPLICIT SUBSTITUTION: Route-deviation magnitude used instead of AIS-gap due to API limits."
        }
    }
    
    if origin_data.get("SYNTHETIC_TEST_DATA"):
        result["SYNTHETIC_TEST_DATA"] = True
        result["WARNING"] = "SCORE IS SYNTHETIC AND FABRICATED. DO NOT USE."
    
    return result
