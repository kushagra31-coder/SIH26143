import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, AlertTriangle, Crosshair, Navigation, Compass, Waves, Wind, ShieldCheck, ExternalLink, MapPin } from 'lucide-react';

export type ExplainerEntity = 'GROUNDING' | 'DEVIATION' | 'PLANNED' | 'HINDCAST' | 'SLICK' | 'WAKASHIO' | null;

interface ExplainerProps {
  entity: ExplainerEntity;
  onClose: () => void;
  onFlyTo?: (target: 'GROUNDING' | 'DEVIATION' | 'PLANNED' | 'HINDCAST' | 'SLICK') => void;
}

interface EntityDetail {
  title: string;
  badge: string;
  badgeColor: string;
  timeStr: string;
  coordsDD: string;
  coordsDMS: string;
  summary: string;
  telemetry: { label: string; value: string }[];
  physicsExplanation: string;
  investigationReport: string;
  confidenceScore?: string;
  actionTarget?: 'GROUNDING' | 'DEVIATION' | 'PLANNED' | 'HINDCAST' | 'SLICK';
}

const DETAILS: Record<Exclude<ExplainerEntity, null>, EntityDetail> = {
  GROUNDING: {
    title: "Ground Truth Grounding Point",
    badge: "CASUALTY ANCHOR",
    badgeColor: "bg-red-500/20 text-accent-coral border-red-500/40",
    timeStr: "2020-07-25 19:25 Local Time (15:25 UTC)",
    coordsDD: "57.7433° E, 20.4433° S",
    coordsDMS: "057°44.6' E, 20°26.6' S",
    summary: "Bulk carrier MV Wakashio struck the barrier reef off Pointe d'Esny, Mauritius at 11.2 knots while on autopilot.",
    telemetry: [
      { label: "Bathymetric Depth", value: "11.5 m (Coral Barrier)" },
      { label: "Impact Speed", value: "11.2 knots" },
      { label: "Heading at Impact", value: "241.0° (Direct to shore)" },
      { label: "Offset to Drift Hindcast", value: "0.62 km (Near exact match)" }
    ],
    physicsExplanation: "The vessel's 18.5-meter loaded draft encountered shallow reef crest bathymetry, causing severe structural buckling in cargo hold #8 and tearing starboard double-bottom bunker fuel tanks.",
    investigationReport: "Panama Maritime Authority Official Report: Bridge team navigated too close to shore to search for cellular Wi-Fi signal during a crew celebration, violating Bridge Resource Management.",
    confidenceScore: "100% (Confirmed Ground Truth)",
    actionTarget: 'GROUNDING'
  },
  DEVIATION: {
    title: "Anomalous Course Deviation Initiation",
    badge: "BEHAVIORAL ANOMALY",
    badgeColor: "bg-amber-500/20 text-accent-amber border-amber-500/40",
    timeStr: "2020-07-25 16:00 Local Time (12:00 UTC)",
    coordsDD: "58.3567° E, 20.1383° S",
    coordsDMS: "058°21.4' E, 20°08.3' S",
    summary: "Chief Officer took navigation watch and altered course from standard open-ocean heading directly toward Mauritius.",
    telemetry: [
      { label: "Course Change", value: "Altered to 241°" },
      { label: "Distance from Coast", value: "65.4 km offshore" },
      { label: "Corridor Excursion", value: "43.32 km off plan" },
      { label: "Vessel Speed", value: "11.4 knots" }
    ],
    physicsExplanation: "Course 241° points straight at Pointe d'Esny headland. The electronic chart system (ECDIS) was not set with appropriate safety depth contours, so audible grounding alarms failed to trigger.",
    investigationReport: "Court of Investigation Records: Captain approved deviation to pass within 5 nautical miles of Mauritius to obtain internet connection for crew phone calls.",
    confidenceScore: "86.64% (Behavioral Evidence)",
    actionTarget: 'DEVIATION'
  },
  PLANNED: {
    title: "Authorized Pilot-to-Pilot Corridor",
    badge: "APPROVED PASSAGE",
    badgeColor: "bg-slate-500/20 text-gray-300 border-slate-500/40",
    timeStr: "Pre-departure Voyage Plan (Singapore -> Tubarao)",
    coordsDD: "Corridor Waypoint 22 to Waypoint 23",
    coordsDMS: "WP 22: 10°00.0'S 078°00.0'E | WP 23: 20°45.0'S 058°00.0'E",
    summary: "The official passage plan filed prior to transit safely bypassed Mauritius by more than 43 kilometers to the south.",
    telemetry: [
      { label: "Corridor Type", value: "Deepwater Pilot-to-Pilot" },
      { label: "Clearance to Coast", value: "43.32 km minimum safe distance" },
      { label: "Sea Room", value: "Open Ocean (>3,500m depth)" },
      { label: "High Risk Area", value: "Outside HRA" }
    ],
    physicsExplanation: "The planned track followed oceanic trade routes utilizing the South Equatorial Current while maintaining immense bathymetric clearance (>3,000 meters depth).",
    investigationReport: "The departure from this corridor had no maritime justification, navigation necessity, or weather avoidance requirement.",
    confidenceScore: "Corridor Verified",
    actionTarget: 'PLANNED'
  },
  HINDCAST: {
    title: "OpenDrift Hindcast Origin Centroid",
    badge: "PHYSICS BACKTRACK",
    badgeColor: "bg-cyan-500/20 text-accent-cyan border-cyan-500/40",
    timeStr: "12-Day Simulation Backtrack (Aug 6 -> July 25)",
    coordsDD: "57.7450° E, 20.4380° S",
    coordsDMS: "057°44.7' E, 20°26.3' S",
    summary: "Lagrangian particle hindcast computed backwards in time from satellite slick detections using CMEMS and GFS currents.",
    telemetry: [
      { label: "Hindcast Model", value: "OpenDrift OceanDrift v1.9" },
      { label: "Ocean Forcing", value: "CMEMS Global Reanalysis (0.38 m/s)" },
      { label: "Windage Factor", value: "3.2% of GFS 10m Wind" },
      { label: "Uncertainty Radius", value: "26.06 km (95% CI)" }
    ],
    physicsExplanation: "Particles are driven backwards by reversing hydrodynamic vectors: x(t - Δt) = x(t) - (u_curr + α u_wind)Δt + η. Turbulent diffusion accounts for bathymetric shear around the coral lagoon.",
    investigationReport: "Spatial match confirmed: The backward trajectory converges directly onto the MV Wakashio grounding anchor within 0.62 km distance.",
    confidenceScore: "98.76% (Spatial Correlation Score)",
    actionTarget: 'HINDCAST'
  },
  SLICK: {
    title: "Sentinel-1 SAR Oil Slick Detection",
    badge: "SATELLITE SEGMENTATION",
    badgeColor: "bg-teal-500/20 text-accent-teal border-teal-500/40",
    timeStr: "2020-08-06 05:43 UTC (Orbital Acquisition)",
    coordsDD: "57.6609° E, 20.5473° S",
    coordsDMS: "057°39.7' E, 20°32.8' S",
    summary: "Copernicus Sentinel-1 C-band Synthetic Aperture Radar detected dark radiometric backscatter anomalies along Mauritius coast.",
    telemetry: [
      { label: "Sensor", value: "C-SAR (VV + VH Dual-Pol)" },
      { label: "Oil Slick Area", value: "12.4 km² Total Plume" },
      { label: "Segmentation Model", value: "Deep Residual U-Net" },
      { label: "Mean Probability", value: "96.3% on primary patch" }
    ],
    physicsExplanation: "Mineral oil films dampen ocean capillary-gravity waves, suppressing Bragg scattering and creating a high-contrast dark surface patch independent of cloud cover.",
    investigationReport: "Satellite footprint matches ground-based disaster reports from the Mauritius Ministry of Environment following the bunker rupture.",
    confidenceScore: "87.4% (ML Detection Confidence)",
    actionTarget: 'SLICK'
  },
  WAKASHIO: {
    title: "MV Wakashio Vessel Dossier",
    badge: "RANK #1 ATTRIBUTION",
    badgeColor: "bg-red-500/20 text-accent-coral border-red-500/40",
    timeStr: "Incident Replay July 25 - August 15, 2020",
    coordsDD: "IMO 9337119 | MMSI 372711000",
    coordsDMS: "Panama Flag | Capesize Bulk Carrier (203,130 DWT)",
    summary: "Capesize bulk carrier owned by Okiyo Maritime / Nagashiki Shipping. Chartered by Mitsui O.S.K. Lines (MOL).",
    telemetry: [
      { label: "Overall Correlation", value: "93.91% Composite Match" },
      { label: "Spatial Proximity", value: "98.76% (0.62 km to hindcast)" },
      { label: "Behavioral Score", value: "86.64% (43.32 km deviation)" },
      { label: "Fuel Oil On Board", value: "3,898 MT VLSFO + 207 MT MGO" }
    ],
    physicsExplanation: "Wakashio is the only vessel exhibiting both temporal co-presence (July 25, 19:25 LT) at the precise hindcast origin and a severe navigational deviation vector leading into the reef.",
    investigationReport: "Captain Sunil Kumar Nandeshwar and Chief Officer Tilakaratna Subodha were arrested and convicted by the Supreme Court of Mauritius.",
    confidenceScore: "93.91% Forensic Attribution Score",
    actionTarget: 'GROUNDING'
  }
};

export default function ForensicExplainer({ entity, onClose, onFlyTo }: ExplainerProps) {
  if (!entity) return null;
  const detail = DETAILS[entity];

  return (
    <AnimatePresence>
      <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
        <motion.div
          initial={{ opacity: 0, scale: 0.95, y: 15 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.95, y: 15 }}
          transition={{ duration: 0.2 }}
          className="bg-navy-900 border border-navy-700 rounded-xl shadow-2xl max-w-2xl w-full overflow-hidden flex flex-col max-h-[85vh]"
        >
          {/* Header */}
          <div className="flex justify-between items-center p-5 border-b border-navy-800 bg-navy-800/60">
            <div className="flex items-center space-x-3">
              <span className={`text-[10px] font-mono px-2.5 py-1 rounded border tracking-widest uppercase font-bold ${detail.badgeColor}`}>
                {detail.badge}
              </span>
              <h2 className="text-lg font-bold text-white tracking-wide">{detail.title}</h2>
            </div>
            <button
              onClick={onClose}
              className="p-1.5 rounded-lg text-gray-400 hover:text-white hover:bg-navy-700 transition-colors"
            >
              <X size={18} />
            </button>
          </div>

          {/* Scrollable Body */}
          <div className="p-6 space-y-6 overflow-y-auto font-sans text-gray-300 text-sm">
            
            {/* Coordinates & Timestamp Bar */}
            <div className="grid grid-cols-2 gap-3 p-3.5 bg-navy-950/80 border border-navy-800 rounded-lg font-mono text-xs">
              <div>
                <span className="text-gray-500 block text-[10px] uppercase tracking-wider mb-0.5">TIMESTAMP</span>
                <span className="text-accent-cyan">{detail.timeStr}</span>
              </div>
              <div>
                <span className="text-gray-500 block text-[10px] uppercase tracking-wider mb-0.5">GEOSPATIAL COORDINATES</span>
                <span className="text-white">{detail.coordsDD}</span>
                <span className="text-gray-400 block text-[10px]">{detail.coordsDMS}</span>
              </div>
            </div>

            {/* Summary */}
            <div>
              <h3 className="text-xs font-mono text-gray-400 uppercase tracking-widest mb-1.5 flex items-center">
                <Compass className="w-3.5 h-3.5 mr-1.5 text-accent-cyan" /> INCIDENT CONTEXT
              </h3>
              <p className="text-gray-200 text-sm leading-relaxed bg-navy-800/40 p-3 rounded border border-navy-700/50">
                {detail.summary}
              </p>
            </div>

            {/* Telemetry Grid */}
            <div>
              <h3 className="text-xs font-mono text-gray-400 uppercase tracking-widest mb-2 flex items-center">
                <Navigation className="w-3.5 h-3.5 mr-1.5 text-accent-amber" /> MARITIME TELEMETRY & MEASUREMENTS
              </h3>
              <div className="grid grid-cols-2 gap-2.5">
                {detail.telemetry.map((t, idx) => (
                  <div key={idx} className="bg-navy-800/60 border border-navy-700/70 p-2.5 rounded">
                    <div className="text-[11px] text-gray-400 font-mono mb-0.5">{t.label}</div>
                    <div className="text-white font-mono font-bold text-xs">{t.value}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Ocean Physics Explanation */}
            <div>
              <h3 className="text-xs font-mono text-gray-400 uppercase tracking-widest mb-1.5 flex items-center">
                <Waves className="w-3.5 h-3.5 mr-1.5 text-accent-teal" /> OCEANOGRAPHIC & PHYSICAL ANALYSIS
              </h3>
              <div className="bg-navy-950/60 border border-teal-500/20 p-3.5 rounded text-xs leading-relaxed text-gray-300">
                {detail.physicsExplanation}
              </div>
            </div>

            {/* Accident Investigation Record */}
            <div>
              <h3 className="text-xs font-mono text-gray-400 uppercase tracking-widest mb-1.5 flex items-center">
                <ShieldCheck className="w-3.5 h-3.5 mr-1.5 text-accent-coral" /> INVESTIGATION RECORD & LEGAL FINDINGS
              </h3>
              <div className="bg-navy-950/60 border border-coral-500/20 p-3.5 rounded text-xs leading-relaxed text-gray-300 italic">
                "{detail.investigationReport}"
              </div>
            </div>

            {/* Confidence Score Pill */}
            {detail.confidenceScore && (
              <div className="flex items-center justify-between p-3 bg-navy-800/80 border border-navy-700 rounded-lg">
                <span className="font-mono text-xs text-gray-400">EVIDENCE CONFIDENCE RATING</span>
                <span className="font-mono text-xs font-bold text-accent-cyan bg-accent-cyan/10 px-2.5 py-1 rounded border border-accent-cyan/30">
                  {detail.confidenceScore}
                </span>
              </div>
            )}
          </div>

          {/* Footer with Actions */}
          <div className="p-4 border-t border-navy-800 bg-navy-950 flex justify-between items-center">
            <span className="text-[11px] font-mono text-gray-500">OCEANTRACE FORENSIC ENGINE</span>
            <div className="flex space-x-2">
              {detail.actionTarget && onFlyTo && (
                <button
                  onClick={() => {
                    onFlyTo(detail.actionTarget!);
                    onClose();
                  }}
                  className="flex items-center space-x-1.5 px-3.5 py-2 rounded bg-navy-800 hover:bg-navy-700 text-accent-cyan text-xs font-mono border border-navy-700 transition-colors"
                >
                  <MapPin size={13} />
                  <span>FOCUS ON MAP</span>
                </button>
              )}
              <button
                onClick={onClose}
                className="px-4 py-2 rounded bg-accent-cyan hover:bg-white text-navy-900 text-xs font-bold font-mono transition-colors"
              >
                CLOSE INSPECTOR
              </button>
            </div>
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  );
}
