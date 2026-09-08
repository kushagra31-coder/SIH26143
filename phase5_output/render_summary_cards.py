import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def render_summary_cards():
    os.makedirs("phase5_output", exist_ok=True)
    
    # -------------------------------------------------------------
    # Card 1: Origin & Location Result (26.06 km offset)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=300)
    fig.patch.set_facecolor('#0F172A')
    ax.set_facecolor('#0F172A')
    ax.axis('off')

    ax.text(0.06, 0.90, 'PHASE 2 DRIFT HINDCAST: ORIGIN RESULT', fontsize=11, fontweight='bold', color='#38BDF8', transform=ax.transAxes)
    ax.text(0.06, 0.82, 'Spatial Backtracking vs Official Ground Truth', fontsize=16, fontweight='bold', color='#FFFFFF', transform=ax.transAxes)

    rect_offset = patches.FancyBboxPatch((0.06, 0.53), 0.88, 0.22, boxstyle='round,pad=0.03', facecolor='#1E293B', edgecolor='#0284C7', linewidth=1.5, transform=ax.transAxes)
    ax.add_patch(rect_offset)

    ax.text(0.10, 0.67, 'PREDICTED SPATIAL OFFSET', fontsize=9.5, fontweight='bold', color='#94A3B8', transform=ax.transAxes)
    ax.text(0.10, 0.57, '26.06 km', fontsize=26, fontweight='bold', color='#38BDF8', transform=ax.transAxes)
    ax.text(0.48, 0.63, 'Convergence: < 30 km oceanographic corridor\nDirect correlation with official casualty coordinates', fontsize=9.5, color='#CBD5E1', transform=ax.transAxes)

    rect_coords = patches.FancyBboxPatch((0.06, 0.16), 0.88, 0.32, boxstyle='round,pad=0.03', facecolor='#1E293B', edgecolor='#334155', linewidth=1.2, transform=ax.transAxes)
    ax.add_patch(rect_coords)

    ax.text(0.10, 0.40, '★ Predicted / Backtracked Origin', fontsize=10.5, fontweight='bold', color='#60A5FA', transform=ax.transAxes)
    ax.text(0.10, 0.33, '57.967° E, -20.548° S', fontsize=13, fontweight='bold', color='#F8FAFC', transform=ax.transAxes)
    ax.text(0.10, 0.27, 'Source: 12-day OpenDrift simulation', fontsize=8.5, color='#94A3B8', transform=ax.transAxes)
    ax.text(0.10, 0.21, 'Timestamp: 2020-07-25 15:25:00 UTC', fontsize=8.5, color='#64748B', transform=ax.transAxes)

    ax.plot([0.52, 0.52], [0.19, 0.44], color='#334155', linewidth=1.2, transform=ax.transAxes)

    ax.text(0.56, 0.40, '✖ Documented Ground Truth', fontsize=10.5, fontweight='bold', color='#F87171', transform=ax.transAxes)
    ax.text(0.56, 0.33, '57.743° E, -20.443° S', fontsize=13, fontweight='bold', color='#F8FAFC', transform=ax.transAxes)
    ax.text(0.56, 0.27, 'Site: Pointe d\'Esny Reef, Mauritius', fontsize=8.5, color='#94A3B8', transform=ax.transAxes)
    ax.text(0.56, 0.21, 'Source: Panama Maritime Authority Report', fontsize=8.5, color='#64748B', transform=ax.transAxes)

    ax.text(0.06, 0.06, '*Note: Hindcast validated using georeferenced seed coordinates prior to full Phase 1 U-Net integration.', fontsize=8, fontstyle='italic', color='#64748B', transform=ax.transAxes)

    plt.tight_layout()
    out1 = "phase5_output/origin_location_result.png"
    plt.savefig(out1, dpi=300, facecolor='#0F172A')
    plt.close()
    print(f"PASS: Generated {out1}")

    # -------------------------------------------------------------
    # Card 2: Vessel Attribution & Evidence Score (63.38/100)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=300)
    fig.patch.set_facecolor('#0F172A')
    ax.set_facecolor('#0F172A')
    ax.axis('off')

    ax.text(0.06, 0.90, 'PHASE 4 EVIDENCE ENGINE: SCORING RESULT', fontsize=11, fontweight='bold', color='#38BDF8', transform=ax.transAxes)
    ax.text(0.06, 0.82, 'Vessel Attribution & Match Confidence', fontsize=16, fontweight='bold', color='#FFFFFF', transform=ax.transAxes)

    rect_meta = patches.FancyBboxPatch((0.06, 0.69), 0.88, 0.10, boxstyle='round,pad=0.02', facecolor='#1E293B', edgecolor='#334155', linewidth=1.2, transform=ax.transAxes)
    ax.add_patch(rect_meta)
    ax.text(0.09, 0.74, 'TARGET: MV Wakashio', fontsize=12, fontweight='bold', color='#F8FAFC', transform=ax.transAxes)
    ax.text(0.42, 0.74, 'IMO: 9337119', fontsize=11.5, fontweight='bold', color='#94A3B8', transform=ax.transAxes)
    ax.text(0.68, 0.74, 'TYPE: Bulk Carrier', fontsize=10.5, fontweight='bold', color='#64748B', transform=ax.transAxes)

    rect_total = patches.FancyBboxPatch((0.06, 0.20), 0.36, 0.45, boxstyle='round,pad=0.03', facecolor='#1E293B', edgecolor='#6366F1', linewidth=1.5, transform=ax.transAxes)
    ax.add_patch(rect_total)
    ax.text(0.10, 0.58, 'TOTAL EVIDENCE SCORE', fontsize=9.5, fontweight='bold', color='#A5B4FC', transform=ax.transAxes)
    ax.text(0.10, 0.44, '63.38', fontsize=32, fontweight='bold', color='#818CF8', transform=ax.transAxes)
    ax.text(0.30, 0.45, '/ 100', fontsize=14, fontweight='bold', color='#6366F1', transform=ax.transAxes)
    ax.text(0.10, 0.33, 'Weighted Score Breakdown:', fontsize=8.5, color='#94A3B8', transform=ax.transAxes)
    ax.text(0.10, 0.27, '• 50% Spatial Correlation\n• 50% Behavioural Deviation', fontsize=8.5, color='#CBD5E1', transform=ax.transAxes)

    rect_breakdown = patches.FancyBboxPatch((0.45, 0.20), 0.49, 0.45, boxstyle='round,pad=0.03', facecolor='#1E293B', edgecolor='#334155', linewidth=1.2, transform=ax.transAxes)
    ax.add_patch(rect_breakdown)

    ax.text(0.48, 0.58, 'Spatial Match', fontsize=11, fontweight='bold', color='#F8FAFC', transform=ax.transAxes)
    ax.text(0.78, 0.58, '47.88 / 100', fontsize=11, fontweight='bold', color='#38BDF8', transform=ax.transAxes)
    bar_bg1 = patches.Rectangle((0.48, 0.53), 0.42, 0.025, facecolor='#334155', transform=ax.transAxes)
    bar_fill1 = patches.Rectangle((0.48, 0.53), 0.42 * 0.4788, 0.025, facecolor='#38BDF8', transform=ax.transAxes)
    ax.add_patch(bar_bg1)
    ax.add_patch(bar_fill1)
    ax.text(0.48, 0.48, 'Origin offset: 26.06 km from grounding site', fontsize=8.5, color='#94A3B8', transform=ax.transAxes)

    ax.text(0.48, 0.38, 'Behaviour Match', fontsize=11, fontweight='bold', color='#F8FAFC', transform=ax.transAxes)
    ax.text(0.78, 0.38, '86.64 / 100', fontsize=11, fontweight='bold', color='#4ADE80', transform=ax.transAxes)
    bar_bg2 = patches.Rectangle((0.48, 0.33), 0.42, 0.025, facecolor='#334155', transform=ax.transAxes)
    bar_fill2 = patches.Rectangle((0.48, 0.33), 0.42 * 0.8664, 0.025, facecolor='#4ADE80', transform=ax.transAxes)
    ax.add_patch(bar_bg2)
    ax.add_patch(bar_fill2)
    ax.text(0.48, 0.28, 'Route deviation: 43.32 km off planned passage', fontsize=8.5, color='#94A3B8', transform=ax.transAxes)

    ax.text(0.06, 0.09, '⚠️ SCOPE BOUNDARY: Single-vessel validation demonstrated. Multi-vessel candidate ranking is pending', fontsize=8.5, fontweight='bold', color='#F59E0B', transform=ax.transAxes)
    ax.text(0.06, 0.05, 'production-tier AIS access (GFW Events API returned 0 bulk carriers in this bounding box).', fontsize=8.5, color='#94A3B8', transform=ax.transAxes)

    plt.tight_layout()
    out2 = "phase5_output/vessel_scoring_result.png"
    plt.savefig(out2, dpi=300, facecolor='#0F172A')
    plt.close()
    print(f"PASS: Generated {out2}")
    return True

if __name__ == "__main__":
    render_summary_cards()
