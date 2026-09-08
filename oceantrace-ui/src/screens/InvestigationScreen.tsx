import React, { useState } from 'react';
import MapViewer from '../components/MapViewer';
import EvidenceDrawer from '../components/EvidenceDrawer';
import type { TimelineState } from '../types/investigation';
import {
  Layers, ChevronLeft, ChevronRight, Eye, EyeOff, Satellite, Waves,
  Navigation, ShieldCheck, Info, Clock, Sliders, BarChart2, X,
  AlertTriangle, CheckCircle2, Circle, Database
} from 'lucide-react';
import clsx from 'clsx';

// ─── Layer Definitions ────────────────────────────────────────────────────────
interface LayerDef {
  id: string;
  label: string;
  description: string;
  variable: string;
  unit: string;
  colorbar: 'oil' | 'current' | 'wind';
  min: string;
  max: string;
  active: boolean;
  visible: boolean;
  color: string;
}

const INITIAL_LAYERS: LayerDef[] = [
  { id: 'slick', label: 'SAR Oil Slick', description: 'Sentinel-1 C-SAR oil spill segmentation (U-Net)', variable: 'Oil Extent', unit: 'km²', colorbar: 'oil', min: '0', max: '12.4', active: true, visible: true, color: '#2dd4bf' },
  { id: 'particles', label: 'Drift Particles (180)', description: 'OpenDrift backward Lagrangian hindcast tracers', variable: 'Particle Density', unit: 'tracers/cell', colorbar: 'current', min: '0', max: '180', active: true, visible: true, color: '#06b6d4' },
  { id: 'actualTrack', label: 'AIS Deviation Track', description: 'MV Wakashio actual vessel trajectory (Course 241°)', variable: 'Vessel Path', unit: '—', colorbar: 'wind', min: 'Deviation', max: 'Grounding', active: true, visible: true, color: '#f87171' },
  { id: 'plannedCorridor', label: 'Planned Safe Corridor', description: 'Authorized pilot-to-pilot passage waypoints 22–23', variable: 'Route Corridor', unit: '—', colorbar: 'wind', min: 'Departure', max: 'Arrival', active: true, visible: true, color: '#94a3b8' },
  { id: 'offsetVector', label: 'Hindcast Offset Vector', description: 'Distance connector between origin centroid and grounding (26.06 km)', variable: 'Spatial Offset', unit: 'km', colorbar: 'oil', min: '0', max: '26.06', active: true, visible: true, color: '#f59e0b' },
];

const STAGES: { id: TimelineState; num: string; label: string; description: string; icon: any }[] = [
  { id: 'SATELLITE', num: '01', label: 'SAR Detection', description: 'Oil slick identified via radar backscatter', icon: Satellite },
  { id: 'DRIFT', num: '02', label: 'Drift Hindcast', description: 'Backward Lagrangian particle simulation', icon: Waves },
  { id: 'AIS', num: '03', label: 'AIS Correlation', description: 'Vessel route deviation analysis', icon: Navigation },
  { id: 'ATTRIBUTION', num: '04', label: 'Attribution', description: 'Evidence-weighted candidate scoring', icon: ShieldCheck },
];

export default function InvestigationScreen() {
  const [timelineState, setTimelineState] = useState<TimelineState>('SATELLITE');
  const [selectedVessel, setSelectedVessel] = useState<string | null>(null);
  const [layers, setLayers] = useState<LayerDef[]>(INITIAL_LAYERS);
  const [leftPanelOpen, setLeftPanelOpen] = useState(true);
  const [rightPanelOpen, setRightPanelOpen] = useState(false);
  const [activePanel, setActivePanel] = useState<'layers' | 'stages' | 'info'>('stages');

  // Active colorbar based on current visible layer
  const activeLayer = layers.find(l => l.visible && l.id === 'slick') || layers[0];

  const toggleLayer = (id: string) => {
    setLayers(prev => prev.map(l => l.id === id ? { ...l, visible: !l.visible } : l));
  };

  const stageIndex = STAGES.findIndex(s => s.id === timelineState);

  return (
    <div className="h-full w-full flex overflow-hidden relative" style={{ background: 'var(--color-cmems-950)' }}>

      {/* ================================================
          LEFT PANEL — Copernicus Layers / Variables Sidebar
          ================================================ */}
      <div
        className="flex-none flex transition-all duration-200 z-20"
        style={{ width: leftPanelOpen ? '300px' : '44px', height: '100%', position: 'relative' }}
      >
        {/* Collapsed Icon Strip */}
        {!leftPanelOpen && (
          <div className="w-full h-full flex flex-col items-center pt-3 gap-3" style={{ background: 'rgba(5,20,38,0.97)', borderRight: '1px solid rgba(59,130,246,0.15)' }}>
            {[
              { key: 'stages', icon: Clock },
              { key: 'layers', icon: Layers },
              { key: 'info', icon: Info },
            ].map(({ key, icon: Icon }) => (
              <button
                key={key}
                onClick={() => { setLeftPanelOpen(true); setActivePanel(key as any); }}
                className="w-8 h-8 rounded-lg flex items-center justify-center transition-colors"
                style={{
                  background: activePanel === key ? 'rgba(6,182,212,0.15)' : 'transparent',
                  color: activePanel === key ? 'var(--color-ocean-cyan)' : 'var(--color-text-muted)',
                  border: activePanel === key ? '1px solid rgba(6,182,212,0.3)' : '1px solid transparent',
                }}
              >
                <Icon size={15} />
              </button>
            ))}
            <button
              onClick={() => setLeftPanelOpen(true)}
              className="mt-auto mb-3 w-8 h-8 rounded-lg flex items-center justify-center"
              style={{ color: 'var(--color-text-muted)' }}
            >
              <ChevronRight size={15} />
            </button>
          </div>
        )}

        {/* Expanded Panel */}
        {leftPanelOpen && (
          <div className="w-full h-full flex flex-col" style={{ background: 'rgba(5,20,38,0.97)', borderRight: '1px solid rgba(59,130,246,0.15)' }}>
            {/* Panel Header */}
            <div className="flex items-center justify-between px-4 py-3" style={{ borderBottom: '1px solid rgba(59,130,246,0.12)', background: 'rgba(2,13,26,0.6)' }}>
              {/* Tab Pills */}
              <div className="flex items-center gap-1">
                {[
                  { key: 'stages', icon: Clock, label: 'Stages' },
                  { key: 'layers', icon: Layers, label: 'Layers' },
                  { key: 'info', icon: Info, label: 'Info' },
                ].map(({ key, icon: Icon, label }) => (
                  <button
                    key={key}
                    onClick={() => setActivePanel(key as any)}
                    className="flex items-center gap-1 px-2.5 py-1.5 rounded-md text-[11px] font-medium transition-all"
                    style={{
                      background: activePanel === key ? 'rgba(6,182,212,0.15)' : 'transparent',
                      color: activePanel === key ? 'var(--color-ocean-cyan)' : 'var(--color-text-muted)',
                      border: activePanel === key ? '1px solid rgba(6,182,212,0.25)' : '1px solid transparent',
                      fontFamily: 'var(--font-mono)',
                    }}
                  >
                    <Icon size={11} />
                    <span>{label}</span>
                  </button>
                ))}
              </div>
              <button onClick={() => setLeftPanelOpen(false)} style={{ color: 'var(--color-text-muted)' }}>
                <ChevronLeft size={15} />
              </button>
            </div>

            {/* Panel Content */}
            <div className="flex-1 overflow-y-auto">

              {/* ── STAGES TAB ── */}
              {activePanel === 'stages' && (
                <div className="p-4 space-y-3">
                  <p className="text-[10px] leading-relaxed" style={{ color: 'var(--color-text-muted)', fontFamily: 'var(--font-mono)' }}>
                    Select an investigation phase to fly the map camera to the relevant geographic area and activate associated data layers.
                  </p>

                  {STAGES.map((stage, idx) => {
                    const Icon = stage.icon;
                    const isActive = timelineState === stage.id;
                    const isPast = idx < stageIndex;
                    return (
                      <button
                        key={stage.id}
                        onClick={() => setTimelineState(stage.id)}
                        className="w-full text-left rounded-lg p-3.5 transition-all duration-150"
                        style={{
                          background: isActive ? 'rgba(6,182,212,0.12)' : 'rgba(10,38,68,0.5)',
                          border: isActive ? '1px solid rgba(6,182,212,0.35)' : '1px solid rgba(59,130,246,0.12)',
                          boxShadow: isActive ? '0 0 16px rgba(6,182,212,0.08) inset' : 'none',
                        }}
                      >
                        <div className="flex items-center gap-3">
                          {/* Status Icon */}
                          <div className="flex-none">
                            {isActive ? (
                              <div className="w-6 h-6 rounded-full flex items-center justify-center" style={{ background: 'var(--color-ocean-cyan)' }}>
                                <Icon size={12} style={{ color: 'var(--color-cmems-950)' }} />
                              </div>
                            ) : isPast ? (
                              <CheckCircle2 size={24} style={{ color: 'rgba(16,185,129,0.6)' }} />
                            ) : (
                              <div className="w-6 h-6 rounded-full flex items-center justify-center" style={{ background: 'rgba(59,130,246,0.08)', border: '1px solid rgba(59,130,246,0.2)' }}>
                                <Icon size={11} style={{ color: 'var(--color-text-muted)' }} />
                              </div>
                            )}
                          </div>
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2">
                              <span className="text-[10px] font-mono" style={{ color: isActive ? 'var(--color-ocean-cyan)' : 'var(--color-text-muted)' }}>PHASE {stage.num}</span>
                            </div>
                            <div className="text-sm font-semibold" style={{ color: isActive ? 'var(--color-text-primary)' : 'var(--color-text-secondary)' }}>{stage.label}</div>
                            <div className="text-[11px] mt-0.5" style={{ color: 'var(--color-text-muted)', fontFamily: 'var(--font-mono)' }}>{stage.description}</div>
                          </div>
                        </div>
                      </button>
                    );
                  })}

                  {/* Case Geometry Quick Stats */}
                  <div className="mt-4 rounded-lg p-3" style={{ background: 'rgba(2,13,26,0.6)', border: '1px solid rgba(59,130,246,0.1)' }}>
                    <div className="text-[10px] font-mono font-semibold tracking-widest uppercase mb-2.5" style={{ color: 'var(--color-text-muted)' }}>Case Geometry</div>
                    {[
                      { label: 'Spill Area (SAR)', value: '12.4 km²', color: 'var(--color-ocean-teal)' },
                      { label: 'Hindcast Offset', value: '26.06 km', color: 'var(--color-ocean-cyan)' },
                      { label: 'Route Deviation', value: '43.32 km', color: 'var(--color-ocean-amber)' },
                      { label: 'Attribution Score', value: '93.91%', color: '#f87171' },
                    ].map(s => (
                      <div key={s.label} className="flex justify-between items-center py-1.5" style={{ borderBottom: '1px solid rgba(59,130,246,0.06)' }}>
                        <span className="text-[11px]" style={{ color: 'var(--color-text-muted)', fontFamily: 'var(--font-mono)' }}>{s.label}</span>
                        <span className="text-[11px] font-bold font-mono" style={{ color: s.color }}>{s.value}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* ── LAYERS TAB ── */}
              {activePanel === 'layers' && (
                <div className="p-4 space-y-2">
                  <p className="text-[10px] leading-relaxed mb-3" style={{ color: 'var(--color-text-muted)', fontFamily: 'var(--font-mono)' }}>
                    Toggle the visibility of individual data layers. Each layer corresponds to a distinct evidence source in the forensic pipeline.
                  </p>

                  {layers.map(layer => (
                    <div key={layer.id} className="rounded-lg p-3 transition-all" style={{ background: layer.visible ? 'rgba(10,38,68,0.7)' : 'rgba(5,20,38,0.4)', border: `1px solid ${layer.visible ? 'rgba(59,130,246,0.2)' : 'rgba(59,130,246,0.07)'}` }}>
                      <div className="flex items-start justify-between gap-2">
                        <div className="flex items-center gap-2 flex-1 min-w-0">
                          <div className="w-2.5 h-2.5 rounded-sm flex-none" style={{ background: layer.visible ? layer.color : 'rgba(90,125,154,0.3)' }} />
                          <div className="min-w-0">
                            <div className="text-xs font-medium truncate" style={{ color: layer.visible ? 'var(--color-text-primary)' : 'var(--color-text-muted)' }}>{layer.label}</div>
                            <div className="text-[10px] truncate mt-0.5" style={{ color: 'var(--color-text-muted)', fontFamily: 'var(--font-mono)' }}>{layer.variable} · {layer.unit}</div>
                          </div>
                        </div>
                        <button onClick={() => toggleLayer(layer.id)} className="flex-none transition-colors p-1" style={{ color: layer.visible ? 'var(--color-ocean-cyan)' : 'var(--color-text-muted)' }}>
                          {layer.visible ? <Eye size={14} /> : <EyeOff size={14} />}
                        </button>
                      </div>
                      {layer.visible && (
                        <div className="mt-2 text-[10px] leading-relaxed" style={{ color: 'var(--color-text-muted)', fontFamily: 'var(--font-mono)', borderTop: '1px solid rgba(59,130,246,0.08)', paddingTop: '6px' }}>
                          {layer.description}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}

              {/* ── INFO TAB ── */}
              {activePanel === 'info' && (
                <div className="p-4 space-y-4">
                  <div className="rounded-lg p-3.5" style={{ background: 'rgba(2,13,26,0.6)', border: '1px solid rgba(59,130,246,0.12)' }}>
                    <div className="text-[10px] font-mono font-semibold tracking-widest uppercase mb-3" style={{ color: 'var(--color-text-muted)' }}>Product Metadata</div>
                    {[
                      { label: 'Product ID', value: 'OT-WAKASHIO-2020-V1' },
                      { label: 'Date Created', value: '2024-12-15' },
                      { label: 'CRS', value: 'WGS84 (EPSG:4326)' },
                      { label: 'Temporal Res.', value: '1 day (12 steps)' },
                      { label: 'Spatial Res.', value: '~0.09° (~10 km)' },
                      { label: 'Format', value: 'GeoJSON / MapLibre GL' },
                    ].map(m => (
                      <div key={m.label} className="flex justify-between items-center py-1.5" style={{ borderBottom: '1px solid rgba(59,130,246,0.06)' }}>
                        <span className="text-[11px]" style={{ color: 'var(--color-text-muted)', fontFamily: 'var(--font-mono)' }}>{m.label}</span>
                        <span className="text-[11px]" style={{ color: 'var(--color-text-secondary)', textAlign: 'right', maxWidth: '140px' }}>{m.value}</span>
                      </div>
                    ))}
                  </div>

                  <div className="rounded-lg p-3.5" style={{ background: 'rgba(239,68,68,0.06)', border: '1px solid rgba(239,68,68,0.15)' }}>
                    <div className="flex items-center gap-2 mb-2">
                      <AlertTriangle size={12} style={{ color: '#f87171' }} />
                      <span className="text-[10px] font-mono font-bold tracking-widest uppercase" style={{ color: '#f87171' }}>Data Notice</span>
                    </div>
                    <p className="text-[11px] leading-relaxed" style={{ color: 'rgba(248,113,113,0.8)', fontFamily: 'var(--font-mono)' }}>
                      Attribution scores are computed for demonstration and educational purposes. Forensic evidence data is derived from publicly available official investigation reports.
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* ================================================
          MAIN MAP AREA (full-bleed)
          ================================================ */}
      <div className="flex-1 relative overflow-hidden">
        <MapViewer timelineState={timelineState} layerVisibilityOverride={
          layers.reduce((acc, l) => ({ ...acc, [l.id]: l.visible }), {} as Record<string, boolean>)
        } />

        {/* ── TOP MAP TOOLBAR (floating) ── */}
        <div className="absolute top-3 left-3 z-10 flex items-center gap-2">
          {/* Stage Quick Switcher Pills */}
          <div className="flex items-center gap-1 rounded-lg px-2 py-1.5" style={{ background: 'rgba(5,20,38,0.92)', border: '1px solid rgba(59,130,246,0.2)', backdropFilter: 'blur(12px)' }}>
            {STAGES.map((stage) => {
              const Icon = stage.icon;
              const isActive = timelineState === stage.id;
              return (
                <button
                  key={stage.id}
                  onClick={() => setTimelineState(stage.id)}
                  title={stage.label}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-[11px] font-medium transition-all"
                  style={{
                    background: isActive ? 'rgba(6,182,212,0.2)' : 'transparent',
                    color: isActive ? 'var(--color-ocean-cyan)' : 'var(--color-text-muted)',
                    border: isActive ? '1px solid rgba(6,182,212,0.3)' : '1px solid transparent',
                    fontFamily: 'var(--font-mono)',
                  }}
                >
                  <Icon size={11} />
                  <span className="hidden sm:inline">{stage.label}</span>
                </button>
              );
            })}
          </div>
        </div>

        {/* ── COLORBAR / LEGEND — Bottom Center (Copernicus style) ── */}
        <div
          className="absolute bottom-3 left-1/2 z-10"
          style={{ transform: 'translateX(-50%)', minWidth: '380px', maxWidth: '520px' }}
        >
          <div className="rounded-xl px-4 py-3" style={{ background: 'rgba(5,20,38,0.94)', border: '1px solid rgba(59,130,246,0.2)', backdropFilter: 'blur(16px)' }}>
            {/* Colorbar label */}
            <div className="flex justify-between items-center mb-2">
              <span className="text-[10px] font-mono font-semibold tracking-widest uppercase" style={{ color: 'var(--color-text-muted)' }}>
                {timelineState === 'SATELLITE' ? 'Oil Slick Probability' :
                  timelineState === 'DRIFT' ? 'Lagrangian Particle Density' :
                  timelineState === 'AIS' ? 'Vessel Route Excursion' :
                  'Evidence Attribution Confidence'}
              </span>
              <span className="text-[10px] font-mono" style={{ color: 'var(--color-text-muted)' }}>
                {timelineState === 'SATELLITE' ? 'U-Net Segmentation Score' :
                  timelineState === 'DRIFT' ? 'OpenDrift OceanDrift v1.9' :
                  timelineState === 'AIS' ? 'km from corridor axis' :
                  'Composite Bayesian Score'}
              </span>
            </div>

            {/* Gradient bar */}
            <div className="relative">
              <div
                className="h-3 w-full rounded-sm"
                style={{
                  background: timelineState === 'SATELLITE'
                    ? 'linear-gradient(to right, #060b14, #0a1a2e, #0e4a6f, #078c8c, #1dbf6b, #c5d82b, #e05c1a, #b51c1c)'
                    : timelineState === 'DRIFT'
                    ? 'linear-gradient(to right, #0c0c2e, #1e4d99, #1e8bc3, #2ec4b6, #7ee8a2, #ffd166, #ef476f)'
                    : timelineState === 'AIS'
                    ? 'linear-gradient(to right, #f0f9ff, #7dd3fc, #0ea5e9, #1d4ed8, #1e1b4b)'
                    : 'linear-gradient(to right, #450a0a, #991b1b, #dc2626, #f97316, #fbbf24, #84cc16, #22c55e)',
                }}
              />
              {/* Tick marks */}
              <div className="flex justify-between mt-1">
                {(timelineState === 'SATELLITE'
                  ? ['0%', '12.5%', '25%', '37.5%', '50%', '62.5%', '75%', '87.5%', '100%']
                  : timelineState === 'DRIFT'
                  ? ['Low', '', '', '', 'Med', '', '', 'High']
                  : timelineState === 'AIS'
                  ? ['On-track', '', '', 'Warning', '', 'Deviated']
                  : ['0', '25', '50', '75', '100']
                ).map((tick, i) => (
                  <span key={i} className="text-[9px]" style={{ color: 'var(--color-text-muted)', fontFamily: 'var(--font-mono)' }}>{tick}</span>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* ── MV WAKASHIO CANDIDATE QUICK BUTTON ── */}
        <div className="absolute top-3 right-3 z-10">
          <button
            onClick={() => { setSelectedVessel('v-1'); setRightPanelOpen(true); }}
            className="flex items-center gap-2 px-3.5 py-2 rounded-lg text-xs font-semibold transition-all"
            style={{
              background: 'rgba(248,113,113,0.12)',
              border: '1px solid rgba(248,113,113,0.3)',
              color: '#f87171',
              backdropFilter: 'blur(12px)',
              fontFamily: 'var(--font-mono)',
            }}
            onMouseEnter={e => { e.currentTarget.style.background = 'rgba(248,113,113,0.2)'; }}
            onMouseLeave={e => { e.currentTarget.style.background = 'rgba(248,113,113,0.12)'; }}
          >
            <ShieldCheck size={13} />
            MV WAKASHIO · 93.91% — VIEW DOSSIER
          </button>
        </div>
      </div>

      {/* ================================================
          RIGHT PANEL — Evidence Dossier Drawer
          ================================================ */}
      <EvidenceDrawer
        isOpen={selectedVessel !== null}
        vesselId={selectedVessel}
        onClose={() => setSelectedVessel(null)}
      />
    </div>
  );
}
