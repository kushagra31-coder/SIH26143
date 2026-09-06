import os
import json
import argparse
from evidence_score import evaluate_vessel

def run_scoring(origin_json, vessel_json, output_json):
    if not os.path.exists(origin_json) or not os.path.exists(vessel_json):
        print("FAIL: Missing input JSON files for scoring.")
        return False
        
    with open(origin_json, 'r') as f:
        origin_data = json.load(f)
        
    with open(vessel_json, 'r') as f:
        vessel_data = json.load(f)
        
    print("--- Phase 4: Single-Vessel Validation Scoring ---")
    print("DISCLAIMER: This system is designed for investigation-prioritisation, not absolute proof of guilt.")
    print("Due to GFW API limitations, we are validating the pipeline against the single known vessel (Wakashio).")
    
    result = evaluate_vessel(origin_data, vessel_data)
    
    print("\n[VALIDATION RESULTS]")
    if result.get("SYNTHETIC_TEST_DATA"):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!                      [SYNTHETIC_TEST_DATA]                        !!!")
        print("!!! THIS SCORE IS COMPUTED AGAINST A FAKED, HARDCODED ORIGIN POINT.   !!!")
        print("!!! DO NOT PRESENT THIS SCORE AS VALIDATED EVIDENCE.                  !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
        
    print(f"Vessel: {result['vessel_name']} (IMO: {result['imo']})")
    print(f"Total Evidence Score: {result['total_score']}/100")
    print(f"  - Spatial Match: {result['spatial_evidence']['score']}/100 ({result['spatial_evidence']['distance_to_origin_km']} km from origin)")
    print(f"  - Behaviour Match: {result['behavior_evidence']['score']}/100 ({result['behavior_evidence']['deviation_from_plan_km']} km deviation from passage plan)")
    print(f"  - {result['behavior_evidence']['note']}")
    
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(result, f, indent=2)
        
    print(f"\nSaved validation record to {output_json}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run single-vessel validation scoring")
    parser.add_argument("--origin", default="phase2_drift/outputs/origin.json", help="Path to OpenDrift origin JSON")
    parser.add_argument("--vessel", default="phase3_ais/wakashio_ground_truth.json", help="Path to Vessel ground truth JSON")
    parser.add_argument("--output", default="phase4_matching/validation_record.json", help="Path to output validation record")
    
    args = parser.parse_args()
    run_scoring(args.origin, args.vessel, args.output)
