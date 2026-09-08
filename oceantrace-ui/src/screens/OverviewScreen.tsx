import React from 'react';
import { Play, Satellite, Waves, Navigation, ShieldAlert, Clock, Database, Globe, ChevronRight, ExternalLink, Zap, BarChart2, ArrowRight } from 'lucide-react';
import { motion } from 'framer-motion';
import validationData from '../data/validation.json';

interface Props { onStart: () => void; }

const stagger = { hidden: {}, show: { transition: { staggerChildren: 0.08 } } };
const fadeUp = { hidden: { opacity: 0, y: 16 }, show: { opacity: 1, y: 0, transition: { duration: 0.4 } } };

export default function OverviewScreen({ onStart }: Props) {
  return (
    <div className="h-full w-full overflow-y-auto" style={{ background: 'var(--color-cmems-950)' }}>
      {/* Hero Banner — Copernicus Style */}
      <div className="relative overflow-hidden" style={{ background: 'linear-gradient(135deg, #020d1a 0%, #051426 40%, #071e38 70%, #0a1e35 100%)', borderBottom: '1px solid rgba(59,130,246,0.15)' }}>
        {/* Grid Pattern Overlay */}
        <div className="absolute inset-0 opacity-[0.03]" style={{ backgroundImage: 'linear-gradient(rgba(59,130,246,1) 1px, transparent 1px), linear-gradient(90deg, rgba(59,130,246,1) 1px, transparent 1px)', backgroundSize: '40px 40px' }} />
        {/* Radial Glow */}
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[900px] h-[400px] rounded-full pointer-events-none" style={{ background: 'radial-gradient(ellipse, rgba(6,182,212,0.06) 0%, transparent 70%)', transform: 'translate(-50%, -30%)' }} />

        <div className="relative max-w-7xl mx-auto px-8 py-10">
          <motion.div variants={stagger} initial="hidden" animate="show" className="flex flex-col gap-6">

            {/* Breadcrumb */}
            <motion.div variants={fadeUp} className="flex items-center space-x-2 text-xs" style={{ color: 'var(--color-text-muted)', fontFamily: 'var(--font-mono)' }}>
              <Globe size={11} />
              <span>OCEANTRACE</span>
              <ChevronRight size={10} />
              <span>Maritime Forensic Intelligence</span>
              <ChevronRight size={10} />
              <span style={{ color: 'var(--color-ocean-cyan)' }}>MV Wakashio Incident 2020</span>
            </motion.div>

            {/* Title Row */}
            <motion.div variants={fadeUp} className="flex items-start justify-between gap-8">
              <div className="flex-1">
                <div className="flex items-center gap-3 mb-3">
                  <span className="cmems-tag" style={{ background: 'rgba(239,68,68,0.1)', color: '#f87171', borderColor: 'rgba(239,68,68,0.3)' }}>
                    OIL SPILL FORENSICS
                  </span>
                  <span className="cmems-tag" style={{ background: 'rgba(6,182,212,0.1)', color: 'var(--color-ocean-cyan)', borderColor: 'rgba(6,182,212,0.3)' }}>
                    HISTORICAL REPLAY
                  </span>
                  <span className="cmems-tag" style={{ background: 'rgba(16,185,129,0.1)', color: 'var(--color-ocean-green)', borderColor: 'rgba(16,185,129,0.3)' }}>
                    VALIDATED
                  </span>
                </div>
                <h1 className="text-3xl font-bold mb-2 leading-tight" style={{ color: 'var(--color-text-primary)', letterSpacing: '-0.01em' }}>
                  MV Wakashio Reef Grounding & Oil Spill
                </h1>
                <p className="text-sm mb-1" style={{ color: 'var(--color-text-secondary)' }}>
                  Pointe d'Esny Barrier Reef, Republic of Mauritius &nbsp;•&nbsp; 20°26.6'S 057°44.6'E &nbsp;•&nbsp; 25 July 2020
                </p>
                <p className="text-xs max-w-3xl leading-relaxed mt-3" style={{ color: 'var(--color-text-muted)', fontFamily: 'var(--font-mono)' }}>
                  Multi-source forensic reconstruction using Copernicus Sentinel-1 SAR satellite imagery, OpenDrift backward Lagrangian hydrodynamic simulation, 
                  and historical AIS vessel transponder analysis. Demonstrates operational-grade source attribution pipeline.
                </p>
              </div>

              {/* Attribution Score Card */}
              <div className="flex-none w-72 rounded-xl p-5" style={{ background: 'rgba(10,38,68,0.85)', border: '1px solid rgba(59,130,246,0.25)', backdropFilter: 'blur(12px)' }}>
                <div className="text-xs font-semibold tracking-wider uppercase mb-3" style={{ color: 'var(--color-text-muted)', fontFamily: 'var(--font-mono)' }}>FORENSIC ATTRIBUTION SCORE</div>
                <div className="text-5xl font-black mb-1" style={{ color: 'var(--color-text-primary)', fontFamily: 'var(--font-mono)', lineHeight: 1 }}>
                  {validationData.total_score}
                  <span className="text-xl font-normal ml-1" style={{ color: 'var(--color-text-muted)' }}>/ 100</span>
                </div>
                <div className="w-full h-1.5 rounded-full mb-3" style={{ background: 'rgba(59,130,246,0.15)' }}>
                  <div className="h-full rounded-full" style={{ width: `${validationData.total_score}%`, background: 'linear-gradient(to right, var(--color-ocean-teal), var(--color-ocean-cyan))' }} />
                </div>
                <div className="text-xs font-bold" style={{ color: 'var(--color-ocean-green)' }}>MV Wakashio (IMO {validationData.imo}) — RANKED #1</div>
                <div className="text-xs mt-1" style={{ color: 'var(--color-text-muted)' }}>Panama Flag &nbsp;•&nbsp; Capesize Bulk Carrier</div>
              </div>
            </motion.div>

            {/* 4 Metric Cards — Copernicus KPI Row */}
            <motion.div variants={fadeUp} className="grid grid-cols-4 gap-4">
              {[
                { icon: Satellite, label: 'SAR Detection Area', value: '12.4 km²', sub: 'Copernicus Sentinel-1', accent: 'var(--color-ocean-teal)', id: '01' },
                { icon: Waves, label: 'Hindcast Offset', value: '26.06 km', sub: '12-Day OpenDrift Backward', accent: 'var(--color-ocean-cyan)', id: '02' },
                { icon: Navigation, label: 'Route Deviation', value: '43.32 km', sub: 'Course 241° Anomalous Turn', accent: 'var(--color-ocean-amber)', id: '03' },
                { icon: ShieldAlert, label: 'Spatial Correlation', value: '0.62 km', sub: 'Hindcast to Grounding Anchor', accent: '#f87171', id: '04' },
              ].map((kpi) => {
                const Icon = kpi.icon;
                return (
                  <div key={kpi.id} className="rounded-lg p-4 relative overflow-hidden" style={{ background: 'rgba(10,38,68,0.6)', border: '1px solid rgba(59,130,246,0.15)' }}>
                    <div className="absolute top-0 left-0 w-0.5 h-full rounded-l-lg" style={{ background: kpi.accent }} />
                    <div className="flex items-center gap-2 mb-3">
                      <Icon size={13} style={{ color: kpi.accent }} />
                      <span className="text-[10px] tracking-wider uppercase" style={{ color: 'var(--color-text-muted)', fontFamily: 'var(--font-mono)' }}>
                        {kpi.label}
                      </span>
                    </div>
                    <div className="text-2xl font-bold mb-0.5" style={{ color: kpi.accent, fontFamily: 'var(--font-mono)' }}>{kpi.value}</div>
                    <div className="text-[11px]" style={{ color: 'var(--color-text-muted)' }}>{kpi.sub}</div>
                  </div>
                );
              })}
            </motion.div>
          </motion.div>
        </div>
      </div>

      {/* Main Content — Two Column Layout like Copernicus Products Page */}
      <div className="max-w-7xl mx-auto px-8 py-8 grid grid-cols-3 gap-8">

        {/* Left Column — Dataset Info & Methodology */}
        <div className="col-span-2 space-y-6">

          {/* Incident Chronology — like Copernicus "Product Description" */}
          <Section title="Incident Chronology & Event Sequence">
            <div className="space-y-0">
              {[
                { date: '25 Jul 2020', time: '16:00 LT', event: 'Unauthorized Course Deviation', detail: 'First Officer alters heading to 241° toward Mauritius. Vessel departs authorized pilot-to-pilot corridor by 43.32 km.', color: 'var(--color-ocean-amber)', tag: 'ANOMALY' },
                { date: '25 Jul 2020', time: '19:25 LT', event: 'Reef Grounding at Pointe d\'Esny', detail: 'MV Wakashio impacts barrier coral reef at 11.2 knots. Hull breach on starboard double-bottom bunker tanks (20°26.6\'S 057°44.6\'E).', color: '#f87171', tag: 'CASUALTY' },
                { date: '26–04 Aug', time: 'Ongoing', event: '12-Day Bunker Seepage Phase', detail: 'Slow VLSFO heavy fuel oil seepage through fractured hull into Blue Bay Marine Park. Tidal flushing transports oil through reef passes.', color: 'var(--color-ocean-teal)', tag: 'DRIFT' },
                { date: '06 Aug 2020', time: '05:43 UTC', event: 'Sentinel-1 SAR Satellite Detection', detail: 'Copernicus Sentinel-1A C-SAR (VV+VH dual-pol) orbital pass detects 12.4 km² oil slick footprint as dark backscatter anomaly.', color: 'var(--color-ocean-cyan)', tag: 'DETECTION' },
              ].map((e, i) => (
                <div key={i} className="flex gap-4 group" style={{ borderLeft: `2px solid rgba(59,130,246,0.12)`, marginLeft: '8px' }}>
                  <div className="relative -ml-[9px] mt-3 flex-none">
                    <div className="w-4 h-4 rounded-full border-2 flex items-center justify-center" style={{ background: 'var(--color-cmems-900)', borderColor: e.color }}>
                      <div className="w-1.5 h-1.5 rounded-full" style={{ background: e.color }} />
                    </div>
                  </div>
                  <div className="pb-6 flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-[10px] font-mono" style={{ color: 'var(--color-text-muted)' }}>{e.date} • {e.time}</span>
                      <span className="cmems-tag" style={{ background: `${e.color}18`, color: e.color, borderColor: `${e.color}40` }}>{e.tag}</span>
                    </div>
                    <div className="text-sm font-semibold mb-1" style={{ color: 'var(--color-text-primary)' }}>{e.event}</div>
                    <div className="text-xs leading-relaxed" style={{ color: 'var(--color-text-muted)', fontFamily: 'var(--font-mono)' }}>{e.detail}</div>
                  </div>
                </div>
              ))}
            </div>
          </Section>

          {/* 4-Phase Methodology — like Copernicus "Dataset Features" */}
          <Section title="4-Phase Algorithmic Pipeline">
            <div className="grid grid-cols-2 gap-4">
              {[
                { num: '01', title: 'Satellite Oil Segmentation', tech: 'Deep Residual U-Net', input: 'Sentinel-1 C-SAR (VV/VH dual-pol)', output: '12.4 km² polygon + 87.4% confidence', icon: Satellite, color: 'var(--color-ocean-teal)' },
                { num: '02', title: 'Backward Drift Hindcast', tech: 'OpenDrift OceanDrift v1.9', input: 'CMEMS Global Reanalysis + GFS Winds', output: '26.06 km uncertainty radius (95% CI)', icon: Waves, color: 'var(--color-ocean-cyan)' },
                { num: '03', title: 'AIS Spatial Correlation', tech: 'Trajectory Matching Algorithm', input: 'Historical AIS Transponder Feed', output: '43.32 km planned corridor violation', icon: Navigation, color: 'var(--color-ocean-amber)' },
                { num: '04', title: 'Evidence Scoring & Attribution', tech: 'Bayesian Composite Evidence', input: 'Spatial (98.76%) + Behavioral (86.64%)', output: 'MV Wakashio: 93.91% Forensic Match', icon: BarChart2, color: '#f87171' },
              ].map(m => {
                const Icon = m.icon;
                return (
                  <div key={m.num} className="cmems-card p-4">
                    <div className="flex items-center gap-2 mb-3">
                      <div className="w-6 h-6 rounded-md flex items-center justify-center" style={{ background: `${m.color}20` }}>
                        <Icon size={13} style={{ color: m.color }} />
                      </div>
                      <span className="text-[10px] font-mono font-bold" style={{ color: m.color }}>PHASE {m.num}</span>
                    </div>
                    <div className="text-sm font-semibold mb-2" style={{ color: 'var(--color-text-primary)' }}>{m.title}</div>
                    <div className="space-y-1.5 text-[11px]" style={{ fontFamily: 'var(--font-mono)' }}>
                      <div className="flex justify-between gap-2">
                        <span style={{ color: 'var(--color-text-muted)' }}>Model:</span>
                        <span style={{ color: 'var(--color-text-secondary)' }}>{m.tech}</span>
                      </div>
                      <div className="flex justify-between gap-2">
                        <span style={{ color: 'var(--color-text-muted)' }}>Input:</span>
                        <span style={{ color: 'var(--color-text-secondary)', textAlign: 'right' }}>{m.input}</span>
                      </div>
                      <div className="pt-1.5 mt-1.5" style={{ borderTop: '1px solid rgba(59,130,246,0.12)' }}>
                        <span className="text-[10px] font-bold" style={{ color: m.color }}>→ {m.output}</span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </Section>
        </div>

        {/* Right Column — Product Card & Quick Access (like Copernicus right sidebar) */}
        <div className="space-y-4">
          {/* CTA Card */}
          <div className="rounded-xl overflow-hidden" style={{ background: 'linear-gradient(135deg, rgba(6,182,212,0.12) 0%, rgba(13,148,136,0.08) 100%)', border: '1px solid rgba(6,182,212,0.25)' }}>
            <div className="p-5">
              <div className="text-xs font-mono font-semibold tracking-widest uppercase mb-2" style={{ color: 'var(--color-ocean-cyan)' }}>OPEN INVESTIGATION</div>
              <div className="text-lg font-bold mb-2" style={{ color: 'var(--color-text-primary)' }}>Interactive Forensic Console</div>
              <div className="text-xs mb-4 leading-relaxed" style={{ color: 'var(--color-text-muted)' }}>
                Access the full map viewer with 12-day drift simulation, clickable markers, and layer controls.
              </div>
              <button
                onClick={onStart}
                className="w-full flex items-center justify-center gap-2 py-3 rounded-lg font-semibold text-sm transition-all duration-200"
                style={{ background: 'var(--color-ocean-cyan)', color: 'var(--color-cmems-950)' }}
                onMouseEnter={e => { e.currentTarget.style.background = '#22d3ee'; e.currentTarget.style.transform = 'translateY(-1px)'; }}
                onMouseLeave={e => { e.currentTarget.style.background = 'var(--color-ocean-cyan)'; e.currentTarget.style.transform = 'none'; }}
              >
                <Play size={14} fill="currentColor" />
                Launch Viewer
              </button>
            </div>
          </div>

          {/* Dataset Metadata Panel — like Copernicus right sidebar */}
          <div className="rounded-xl" style={{ background: 'rgba(10,38,68,0.7)', border: '1px solid rgba(59,130,246,0.15)' }}>
            <div className="px-4 py-3" style={{ borderBottom: '1px solid rgba(59,130,246,0.1)' }}>
              <div className="text-xs font-mono font-semibold tracking-widest uppercase" style={{ color: 'var(--color-text-muted)' }}>Dataset Information</div>
            </div>
            <div className="p-4 space-y-3">
              {[
                { label: 'Product ID', value: 'OT-WAKASHIO-2020-V1' },
                { label: 'Data Sources', value: 'Sentinel-1 SAR · CMEMS Global · GFS' },
                { label: 'Temporal Coverage', value: '2020-07-25 → 2020-08-06' },
                { label: 'Spatial Domain', value: '57°E–58.5°E · 19.7°S–21°S' },
                { label: 'Coordinate System', value: 'WGS 84 (EPSG:4326)' },
                { label: 'Drift Particles', value: '180 Lagrangian Tracers' },
                { label: 'Time Steps', value: '13 (Daily · 12-Day Span)' },
                { label: 'Attribution Score', value: `${validationData.total_score}% (Composite Evidence)` },
              ].map(({ label, value }) => (
                <div key={label} className="flex justify-between gap-3 text-xs" style={{ borderBottom: '1px solid rgba(59,130,246,0.06)', paddingBottom: '8px' }}>
                  <span style={{ color: 'var(--color-text-muted)', fontFamily: 'var(--font-mono)' }}>{label}</span>
                  <span style={{ color: 'var(--color-text-secondary)', textAlign: 'right', maxWidth: '160px' }}>{value}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Related Data Products */}
          <div className="rounded-xl" style={{ background: 'rgba(10,38,68,0.7)', border: '1px solid rgba(59,130,246,0.15)' }}>
            <div className="px-4 py-3" style={{ borderBottom: '1px solid rgba(59,130,246,0.1)' }}>
              <div className="text-xs font-mono font-semibold tracking-widest uppercase" style={{ color: 'var(--color-text-muted)' }}>Related Data Sources</div>
            </div>
            <div className="p-3 space-y-2">
              {[
                { label: 'CMEMS Global Ocean Physics', code: 'GLOBAL_PHY_001_024', color: 'var(--color-ocean-cyan)' },
                { label: 'Sentinel-1 SAR Level-2 OCN', code: 'SENTINEL-1_SAR_OCN', color: 'var(--color-ocean-teal)' },
                { label: 'GFS 0.25° Atmospheric Model', code: 'NOAA-GFS-0P25', color: 'var(--color-ocean-amber)' },
              ].map(d => (
                <div key={d.code} className="cmems-card p-3 flex items-center gap-2">
                  <Database size={12} style={{ color: d.color, flexShrink: 0 }} />
                  <div className="flex-1 min-w-0">
                    <div className="text-xs font-medium truncate" style={{ color: 'var(--color-text-secondary)' }}>{d.label}</div>
                    <div className="text-[10px] font-mono" style={{ color: 'var(--color-text-muted)' }}>{d.code}</div>
                  </div>
                  <ExternalLink size={10} style={{ color: 'var(--color-text-muted)', flexShrink: 0 }} />
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-xl overflow-hidden" style={{ background: 'rgba(10,38,68,0.5)', border: '1px solid rgba(59,130,246,0.12)' }}>
      <div className="px-5 py-3.5" style={{ borderBottom: '1px solid rgba(59,130,246,0.1)', background: 'rgba(5,20,38,0.4)' }}>
        <h2 className="text-xs font-semibold tracking-widest uppercase" style={{ color: 'var(--color-text-secondary)', fontFamily: 'var(--font-mono)' }}>{title}</h2>
      </div>
      <div className="p-5">{children}</div>
    </div>
  );
}
