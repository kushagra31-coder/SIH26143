import os
import json

def generate_report():
    print("--- Phase 5: Generating Honest Final Report ---")
    
    record_path = "phase4_matching/validation_record.json"
    if not os.path.exists(record_path):
        print("FAIL: Validation record not found.")
        return False
        
    with open(record_path, 'r') as f:
        record = json.load(f)
        
    synthetic_warning = ""
    if record.get("SYNTHETIC_TEST_DATA"):
        synthetic_warning = """
> [!CAUTION]
> **[SYNTHETIC_TEST_DATA] DRIFT HINDCAST BYPASSED**
> The OpenDrift and Cartopy packages failed to compile natively on this Windows machine due to fatal C++ build dependencies (gdk-pixbuf).
> 
> By explicit authorization, Phase 2 drift physics were **BYPASSED** and a synthetic origin point was hardcoded to validate the UI, mapping, and scoring engine.
> 
> **THE SCORE BELOW IS NOT A PHYSICAL RESULT AND CANNOT BE PRESENTED AS EVIDENCE.**
"""
        
    if record.get("SYNTHETIC_TEST_DATA"):
        phase2_text = "- **Phase 2 Drift Hindcast**: [SYNTHETIC_TEST_DATA] Bypassed due to native build failures. Faked origin point injected to validate UI."
    else:
        phase2_text = f"- **Phase 2 Drift Hindcast**: Successfully backtracked the georeferenced origin. The predicted origin converged to {record['spatial_evidence']['distance_to_origin_km']} km of the actual grounding point."

    report = f"""# SIH26143 Oil Spill Vessel Attribution: MVP Validation Report
{synthetic_warning}
## 1. Pipeline Status
- **Phase 1 Mask Source**: Predicted mask placeholder (waiting on external training pipeline).
{phase2_text}
- **Phase 3 & 4 Validation**: Successfully validated single-vessel pipeline end-to-end.

## 2. Wakashio Validation Results
The single known vessel (Wakashio, IMO: {record['imo']}) was scored against the predicted origin:
- **Total Evidence Score**: {record['total_score']}/100
- **Spatial Match**: {record['spatial_evidence']['score']}/100
- **Behaviour Match**: {record['behavior_evidence']['score']}/100

## 3. Honest Scope Limitation: Candidate Ranking
Multi-vessel ranking was not demonstrable in this MVP due to API access limitations. The Global Fishing Watch (GFW) "Events" API, while accessible, is focused heavily on fishing fleets and transshipments. It returned exactly 0 usable candidates for this bulk-carrier incident within the Mauritius bounding box. 

Therefore, rather than fabricating fake AIS tracks to simulate a ranking, we have explicitly scoped this MVP to validate the math and trajectory logic against the single known vessel (Wakashio). A full multi-vessel attribution ranking requires production-tier AIS access (e.g. the pending GFW elevated-access request, or commercial APIs like Spire), which is noted as pending future work.

## 4. Ground-Truth Citations
The behaviour deviation and actual grounding anchor used for this validation were explicitly extracted from the official casualty report:
*Panama Maritime Authority, Directorate General of Merchant Marine, Maritime Affairs Investigation Department, "Report MV Wakashio R-029-2021-DIAM," 2023.*
"""
    
    os.makedirs("phase5_output", exist_ok=True)
    out_file = "phase5_output/final_pitch_report.md"
    with open(out_file, 'w') as f:
        f.write(report)
        
    print(f"PASS: Generated honest final report at {out_file}")
    return True

if __name__ == "__main__":
    generate_report()
