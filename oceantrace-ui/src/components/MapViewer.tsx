import React, { useMemo, useEffect, useRef, useState, useCallback } from 'react';
import Map, { Source, Layer, Popup, Marker, NavigationControl, ScaleControl } from 'react-map-gl/maplibre';
import 'maplibre-gl/dist/maplibre-gl.css';
import { Play, Pause, SkipBack, SkipForward, Layers, Eye, EyeOff, Info, Compass, Wind, Waves, Navigation, Crosshair, AlertTriangle } from 'lucide-react';
import type { TimelineState } from '../types/investigation';

import predictionsData from '../data/predictions.json';
import originData from '../data/origin.json';
import candidatesData from '../data/candidates.json';
import { PREGENERATED_PARTICLES, DRIFT_TIMESTEPS, type TimeStep } from '../data/driftSimulation';
import ForensicExplainer, { type ExplainerEntity } from './ForensicExplainer';

const MAP_STYLE = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json";

// Robust coordinate parser for DMS and Decimal degrees
function parseCoordinate(val: string | number): number {
  if (typeof val === 'number') return val;
  if (!val) return 0;
  try {
    const isSouth = val.includes('S');
    const isWest = val.includes('W');
    const parts = val.replace(/[^0-9.]/g, ' ').trim().split(/\s+/);
    if (parts.length >= 2) {
      const deg = parseFloat(parts[0]);
      const min = parseFloat(parts[1]);
      const dd = deg + (min / 60);
      return (isSouth || isWest) ? -dd : dd;
    } else if (parts.length === 1) {
      const num = parseFloat(parts[0]);
      return (isSouth || isWest) ? -Math.abs(num) : num;
    }
  } catch {
    return 0;
  }
  return 0;
}

export default function MapViewer({ timelineState, layerVisibilityOverride }: { timelineState: TimelineState; layerVisibilityOverride?: Record<string, boolean> }) {
  const mapRef = useRef<any>(null);

  // Simulation State
  const [simulationStep, setSimulationStep] = useState<number>(0);
  const [isPlayingSimulation, setIsPlayingSimulation] = useState<boolean>(false);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1000); // ms per step

  // Explainer Modal State
  const [activeExplainer, setActiveExplainer] = useState<ExplainerEntity>(null);

  // Quick Hover/Click Popup State
  const [popupInfo, setPopupInfo] = useState<{ lng: number; lat: number; title: string; subtitle: string; entity: ExplainerEntity } | null>(null);

  // Live Cursor Telemetry State
  const [cursorTelemetry, setCursorTelemetry] = useState<{
    lng: string;
    lat: string;
    dms: string;
    depth: string;
  }>({
    lng: "57.7433° E",
    lat: "20.4433° S",
    dms: "057°44.6'E 20°26.6'S",
    depth: "12 m (Coral Barrier)"
  });

  // Layer Visibility Toggles
  const [layerVisibility, setLayerVisibility] = useState({
    slick: true,
    particles: true,
    actualTrack: true,
    plannedCorridor: true,
    offsetVector: true,
    markers: true
  });
  const [showLayerMenu, setShowLayerMenu] = useState(false);

  // 1. Process GeoJSON Data
  const slickGeoJSON = useMemo(() => {
    return {
      type: 'FeatureCollection',
      features: predictionsData.features
    };
  }, []);

  const originGeoJSON = useMemo(() => {
    return {
      type: 'FeatureCollection',
      features: [
        {
          type: 'Feature',
          geometry: {
            type: 'Point',
            coordinates: [originData.origin_lon, originData.origin_lat]
          },
          properties: {}
        }
      ]
    };
  }, []);

  // Distance Connector Vector (Grounding to Hindcast Centroid)
  const offsetConnectorGeoJSON = useMemo(() => {
    return {
      type: 'FeatureCollection',
      features: [
        {
          type: 'Feature',
          geometry: {
            type: 'LineString',
            coordinates: [
              [57.7433, -20.4433], // Grounding Point
              [originData.origin_lon, originData.origin_lat] // Predicted Origin
            ]
          },
          properties: { label: 'Offset: 26.06 km' }
        }
      ]
    };
  }, []);

  // Dynamic 180 Particles at the Current TimeStep
  const particlesGeoJSON = useMemo(() => {
    const features = PREGENERATED_PARTICLES.map((p) => {
      const coords = p.trajectory[simulationStep] || p.trajectory[p.trajectory.length - 1];
      return {
        type: 'Feature',
        geometry: {
          type: 'Point',
          coordinates: coords
        },
        properties: { id: p.id }
      };
    });
    return { type: 'FeatureCollection', features };
  }, [simulationStep]);

  // Actual AIS Deviation Track
  const actualTrackGeoJSON = useMemo(() => {
    const coords = candidatesData.actual_deviation_track.map((pt: any) => {
      return [parseCoordinate(pt.lon), parseCoordinate(pt.lat)];
    });
    return {
      type: 'FeatureCollection',
      features: [
        {
          type: 'Feature',
          geometry: { type: 'LineString', coordinates: coords },
          properties: { type: 'top-candidate' }
        }
      ]
    };
  }, []);

  // Planned Deepwater Passage Corridor
  const plannedTrackGeoJSON = useMemo(() => {
    const coords = candidatesData.planned_passage.map((pt: any) => {
      return [parseCoordinate(pt.lon), parseCoordinate(pt.lat)];
    });
    return {
      type: 'FeatureCollection',
      features: [
        {
          type: 'Feature',
          geometry: { type: 'LineString', coordinates: coords },
          properties: { type: 'planned' }
        }
      ]
    };
  }, []);

  // 2. Camera Animations when TimelineState Changes
  useEffect(() => {
    const map = mapRef.current?.getMap();
    if (!map) return;

    if (timelineState === 'SATELLITE') {
      map.flyTo({ center: [57.66088, -20.5473], zoom: 11.8, pitch: 20, duration: 2500, essential: true });
    } else if (timelineState === 'DRIFT') {
      map.flyTo({ center: [57.72, -20.48], zoom: 11.2, pitch: 30, duration: 2500, essential: true });
      setIsPlayingSimulation(true); // Automatically trigger drift simulation playback
    } else if (timelineState === 'AIS') {
      map.flyTo({ center: [58.05, -20.30], zoom: 9.8, pitch: 25, duration: 2500, essential: true });
      setIsPlayingSimulation(false);
    } else if (timelineState === 'ATTRIBUTION') {
      map.flyTo({ center: [57.74, -20.42], zoom: 11.5, pitch: 35, duration: 2500, essential: true });
      setIsPlayingSimulation(false);
    }
  }, [timelineState]);

  // 3. Dynamic Simulation Playback Timer
  useEffect(() => {
    if (!isPlayingSimulation) return;

    const interval = setInterval(() => {
      setSimulationStep((prev) => {
        if (prev >= 12) {
          return 0; // loop back to start
        }
        return prev + 1;
      });
    }, playbackSpeed);

    return () => clearInterval(interval);
  }, [isPlayingSimulation, playbackSpeed]);

  // 4. Live Mouse Telemetry Update
  const handleMouseMove = useCallback((e: any) => {
    if (!e.lngLat) return;
    const { lng, lat } = e.lngLat;
    const absLat = Math.abs(lat);
    const absLng = Math.abs(lng);

    const latDeg = Math.floor(absLat);
    const latMin = ((absLat - latDeg) * 60).toFixed(1);
    const lngDeg = Math.floor(absLng);
    const lngMin = ((absLng - lngDeg) * 60).toFixed(1);

    const dmsStr = `${String(lngDeg).padStart(3, '0')}°${lngMin}'E ${latDeg}°${latMin}'S`;

    // Bathymetry estimate based on proximity to Mauritius barrier reef (~57.74, -20.44)
    const distToReef = Math.hypot(lng - 57.7433, lat - (-20.4433)) * 111.32; // rough km
    let depthStr = ">3,400 m (Abyssal Basin)";
    if (distToReef < 2.0) {
      depthStr = "11-18 m (Fringing Coral Barrier)";
    } else if (distToReef < 6.0) {
      depthStr = "80-250 m (Continental Shelf)";
    } else if (distToReef < 15.0) {
      depthStr = "1,200 m (Oceanic Slope)";
    }

    setCursorTelemetry({
      lng: `${lng.toFixed(4)}° E`,
      lat: `${absLat.toFixed(4)}° S`,
      dms: dmsStr,
      depth: depthStr
    });
  }, []);

  // 5. Camera FlyTo Target from Explainer Modal
  const handleFlyToTarget = (target: 'GROUNDING' | 'DEVIATION' | 'PLANNED' | 'HINDCAST' | 'SLICK') => {
    const map = mapRef.current?.getMap();
    if (!map) return;

    if (target === 'GROUNDING') {
      map.flyTo({ center: [57.7433, -20.4433], zoom: 13.5, pitch: 45, duration: 2000 });
    } else if (target === 'DEVIATION') {
      map.flyTo({ center: [58.3567, -20.1383], zoom: 11.5, pitch: 30, duration: 2000 });
    } else if (target === 'PLANNED') {
      map.flyTo({ center: [58.0, -20.55], zoom: 9.5, pitch: 10, duration: 2000 });
    } else if (target === 'HINDCAST') {
      map.flyTo({ center: [57.745, -20.438], zoom: 12.5, pitch: 35, duration: 2000 });
    } else if (target === 'SLICK') {
      map.flyTo({ center: [57.66088, -20.5473], zoom: 12.8, pitch: 30, duration: 2000 });
    }
  };

  const currentStepData: TimeStep = DRIFT_TIMESTEPS[simulationStep] || DRIFT_TIMESTEPS[0];

  return (
    <div className="absolute inset-0 w-full h-full select-none overflow-hidden bg-[#040A14]">
      <Map
        ref={mapRef}
        initialViewState={{
          longitude: 57.74,
          latitude: -20.44,
          zoom: 11.2
        }}
        mapStyle={MAP_STYLE}
        interactive={true}
        onMouseMove={handleMouseMove}
        cursor="crosshair"
      >
        <NavigationControl position="top-left" showCompass={true} />
        <ScaleControl position="bottom-left" unit="nautical" />

        {/* ============================================================ */}
        {/* 1. PLANNED CORRIDOR                                         */}
        {/* ============================================================ */}
        {layerVisibility.plannedCorridor && (
          <Source id="planned-route-src" type="geojson" data={plannedTrackGeoJSON as any}>
            <Layer 
              id="planned-route-glow"
              type="line"
              paint={{
                'line-color': '#94A3B8',
                'line-width': 4,
                'line-opacity': 0.3,
                'line-blur': 2
              }}
            />
            <Layer 
              id="planned-route"
              type="line"
              paint={{
                'line-color': '#CBD5E1',
                'line-width': 2,
                'line-dasharray': [3, 2]
              }}
            />
          </Source>
        )}

        {/* ============================================================ */}
        {/* 2. DISTANCE CONNECTOR (OFFSET: 26.06 km)                     */}
        {/* ============================================================ */}
        {layerVisibility.offsetVector && (
          <Source id="offset-connector-src" type="geojson" data={offsetConnectorGeoJSON as any}>
            <Layer 
              id="offset-line"
              type="line"
              paint={{
                'line-color': '#F59E0B',
                'line-width': 2,
                'line-dasharray': [2, 2],
                'line-opacity': 0.85
              }}
            />
          </Source>
        )}

        {/* ============================================================ */}
        {/* 3. OPENDRIFT UNCERTAINTY REGION (26.06 km RADIUS)           */}
        {/* ============================================================ */}
        <Source id="origin-region-src" type="geojson" data={originGeoJSON as any}>
          <Layer 
            id="origin-region"
            type="circle"
            paint={{
              'circle-radius': 95,
              'circle-color': '#FBBF24',
              'circle-opacity': 0.12,
              'circle-blur': 0.8
            }}
          />
          <Layer 
            id="origin-center"
            type="circle"
            paint={{
              'circle-radius': 7,
              'circle-color': '#22D3EE',
              'circle-stroke-width': 2.5,
              'circle-stroke-color': '#ffffff'
            }}
          />
        </Source>

        {/* ============================================================ */}
        {/* 4. DYNAMIC 12-DAY DRIFT PARTICLES (180 PARTICLES)           */}
        {/* ============================================================ */}
        {layerVisibility.particles && (
          <Source id="particles-src" type="geojson" data={particlesGeoJSON as any}>
            <Layer 
              id="particles-glow"
              type="circle"
              paint={{
                'circle-radius': 6,
                'circle-color': '#22D3EE',
                'circle-opacity': 0.25,
                'circle-blur': 1.0
              }}
            />
            <Layer 
              id="particles-core"
              type="circle"
              paint={{
                'circle-radius': 3.5,
                'circle-color': '#22D3EE',
                'circle-opacity': 0.85,
                'circle-stroke-width': 0.7,
                'circle-stroke-color': '#FFFFFF'
              }}
            />
          </Source>
        )}

        {/* ============================================================ */}
        {/* 5. ACTUAL AIS DEVIATION TRACK                                */}
        {/* ============================================================ */}
        {layerVisibility.actualTrack && (
          <Source id="actual-route-src" type="geojson" data={actualTrackGeoJSON as any}>
            <Layer 
              id="actual-route-glow"
              type="line"
              paint={{
                'line-color': '#FB7185',
                'line-width': 8,
                'line-opacity': 0.4,
                'line-blur': 4
              }}
            />
            <Layer 
              id="actual-route"
              type="line"
              paint={{
                'line-color': '#FB7185',
                'line-width': 3,
                'line-opacity': 1.0
              }}
            />
          </Source>
        )}

        {/* ============================================================ */}
        {/* 6. OIL SLICK POLYGON (SENTINEL-1 SAR DETECTION)             */}
        {/* ============================================================ */}
        {layerVisibility.slick && (
          <Source id="slick-src" type="geojson" data={slickGeoJSON as any}>
            <Layer 
              id="slick-fill"
              type="fill"
              paint={{
                'fill-color': '#2DD4BF',
                'fill-opacity': 0.28
              }}
            />
            <Layer 
              id="slick-glow"
              type="line"
              paint={{
                'line-color': '#2DD4BF',
                'line-width': 6,
                'line-opacity': 0.45,
                'line-blur': 4
              }}
            />
            <Layer 
              id="slick-outline"
              type="line"
              paint={{
                'line-color': '#2DD4BF',
                'line-width': 2.2
              }}
            />
          </Source>
        )}

        {/* ============================================================ */}
        {/* 7. INTERACTIVE CLICKABLE MARKERS ON MAP                     */}
        {/* ============================================================ */}
        {layerVisibility.markers && (
          <>
            {/* PIN 1: GROUNDING POINT (Pointe d'Esny) */}
            <Marker
              longitude={57.7433}
              latitude={-20.4433}
              anchor="center"
              onClick={() => setActiveExplainer('GROUNDING')}
            >
              <div className="relative group cursor-pointer flex flex-col items-center">
                <div className="w-5 h-5 rounded-full bg-red-500 flex items-center justify-center text-white text-[10px] font-bold shadow-lg shadow-red-500/50 relative z-10 border-2 border-white">
                  !
                </div>
                <div className="absolute inset-0 w-5 h-5 rounded-full bg-red-500/50 animate-ping-slow pointer-events-none" />
                <div className="mt-1 px-2 py-0.5 rounded bg-navy-950/90 border border-red-500/60 text-accent-coral font-mono text-[10px] font-bold tracking-wider whitespace-nowrap shadow-lg flex items-center space-x-1">
                  <span>GROUNDING (19:25 LT)</span>
                </div>
              </div>
            </Marker>

            {/* PIN 2: DEVIATION INITIATION (16:00 LT) */}
            <Marker
              longitude={58.3567}
              latitude={-20.1383}
              anchor="center"
              onClick={() => setActiveExplainer('DEVIATION')}
            >
              <div className="relative group cursor-pointer flex flex-col items-center">
                <div className="w-4 h-4 rounded-full bg-amber-400 flex items-center justify-center text-navy-950 text-[9px] font-extrabold shadow-lg shadow-amber-500/50 relative z-10 border border-white">
                  ▶
                </div>
                <div className="mt-1 px-2 py-0.5 rounded bg-navy-950/90 border border-amber-400/60 text-accent-amber font-mono text-[10px] font-bold tracking-wider whitespace-nowrap shadow-lg">
                  COURSE 241° (16:00 LT)
                </div>
              </div>
            </Marker>

            {/* PIN 3: OPENDRIFT HINDCAST CENTROID */}
            <Marker
              longitude={originData.origin_lon}
              latitude={originData.origin_lat}
              anchor="center"
              onClick={() => setActiveExplainer('HINDCAST')}
            >
              <div className="relative group cursor-pointer flex flex-col items-center">
                <div className="w-4 h-4 rounded-full bg-cyan-400 flex items-center justify-center text-navy-950 text-[9px] font-extrabold shadow-lg shadow-cyan-400/50 relative z-10 border border-white">
                  ★
                </div>
                <div className="mt-1 px-2 py-0.5 rounded bg-navy-950/90 border border-cyan-400/60 text-accent-cyan font-mono text-[10px] font-bold tracking-wider whitespace-nowrap shadow-lg">
                  HINDCAST CENTROID
                </div>
              </div>
            </Marker>

            {/* PIN 4: SENTINEL-1 SAR SLICK FOOTPRINT */}
            <Marker
              longitude={57.66088}
              latitude={-20.54734}
              anchor="center"
              onClick={() => setActiveExplainer('SLICK')}
            >
              <div className="relative group cursor-pointer flex flex-col items-center">
                <div className="w-4 h-4 rounded-full bg-teal-400 flex items-center justify-center text-navy-950 text-[9px] font-extrabold shadow-lg shadow-teal-400/50 relative z-10 border border-white">
                  S1
                </div>
                <div className="mt-1 px-2 py-0.5 rounded bg-navy-950/90 border border-teal-400/60 text-accent-teal font-mono text-[10px] font-bold tracking-wider whitespace-nowrap shadow-lg">
                  SAR SLICK: 12.4 km²
                </div>
              </div>
            </Marker>
          </>
        )}
      </Map>

      {/* ============================================================ */}
      {/* FLOATING MARITIME HUD OVERLAYS                                */}
      {/* ============================================================ */}

      {/* Top-Right: Live Telemetry & Oceanographic HUD */}
      <div className="absolute top-4 right-4 z-20 flex flex-col items-end space-y-2 pointer-events-auto">
        <div className="bg-navy-950/90 backdrop-blur-md border border-navy-700/80 rounded-lg p-3.5 shadow-2xl min-w-[260px] font-mono text-xs">
          <div className="flex items-center justify-between border-b border-navy-800 pb-2 mb-2">
            <span className="text-gray-400 text-[10px] tracking-widest uppercase flex items-center">
              <Compass className="w-3.5 h-3.5 mr-1 text-accent-cyan" /> OCEAN TELEMETRY
            </span>
            <span className="w-2 h-2 rounded-full bg-accent-teal animate-pulse" />
          </div>

          <div className="space-y-1.5 text-[11px]">
            <div className="flex justify-between">
              <span className="text-gray-500">CURSOR POS:</span>
              <span className="text-accent-cyan font-semibold">{cursorTelemetry.dms}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">BATHYMETRY:</span>
              <span className="text-gray-300">{cursorTelemetry.depth}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">OCEAN CURRENT:</span>
              <span className="text-accent-teal font-semibold">{currentStepData.currentSpeedMs} m/s @ {currentStepData.currentDirection}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">SURFACE WIND:</span>
              <span className="text-accent-amber font-semibold">{currentStepData.windSpeedKts} kts @ {currentStepData.windDirection}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">HINDCAST OFFSET:</span>
              <span className="text-accent-coral font-bold">26.06 km</span>
            </div>
          </div>
        </div>

        {/* Layer Toggle Button & Dropdown */}
        <div className="relative">
          <button
            onClick={() => setShowLayerMenu(!showLayerMenu)}
            className="bg-navy-950/90 hover:bg-navy-800 border border-navy-700 text-gray-300 hover:text-white px-3 py-2 rounded-lg text-xs font-mono flex items-center space-x-2 shadow-lg transition-colors"
          >
            <Layers size={14} className="text-accent-cyan" />
            <span>MAP LAYERS</span>
          </button>

          {showLayerMenu && (
            <div className="absolute right-0 mt-2 w-56 bg-navy-950 border border-navy-700 rounded-lg shadow-2xl p-2.5 space-y-1 text-xs font-mono z-30">
              <span className="text-gray-500 text-[10px] tracking-wider uppercase block px-2 pb-1 border-b border-navy-800">TOGGLE LAYERS</span>
              
              <label className="flex items-center justify-between px-2 py-1.5 hover:bg-navy-800/60 rounded cursor-pointer">
                <span>SAR Oil Slick</span>
                <input
                  type="checkbox"
                  checked={layerVisibility.slick}
                  onChange={(e) => setLayerVisibility({ ...layerVisibility, slick: e.target.checked })}
                  className="accent-accent-cyan"
                />
              </label>
              <label className="flex items-center justify-between px-2 py-1.5 hover:bg-navy-800/60 rounded cursor-pointer">
                <span>Drift Particles (180)</span>
                <input
                  type="checkbox"
                  checked={layerVisibility.particles}
                  onChange={(e) => setLayerVisibility({ ...layerVisibility, particles: e.target.checked })}
                  className="accent-accent-cyan"
                />
              </label>
              <label className="flex items-center justify-between px-2 py-1.5 hover:bg-navy-800/60 rounded cursor-pointer">
                <span>Actual AIS Track</span>
                <input
                  type="checkbox"
                  checked={layerVisibility.actualTrack}
                  onChange={(e) => setLayerVisibility({ ...layerVisibility, actualTrack: e.target.checked })}
                  className="accent-accent-cyan"
                />
              </label>
              <label className="flex items-center justify-between px-2 py-1.5 hover:bg-navy-800/60 rounded cursor-pointer">
                <span>Planned Passage</span>
                <input
                  type="checkbox"
                  checked={layerVisibility.plannedCorridor}
                  onChange={(e) => setLayerVisibility({ ...layerVisibility, plannedCorridor: e.target.checked })}
                  className="accent-accent-cyan"
                />
              </label>
              <label className="flex items-center justify-between px-2 py-1.5 hover:bg-navy-800/60 rounded cursor-pointer">
                <span>Interactive Pins</span>
                <input
                  type="checkbox"
                  checked={layerVisibility.markers}
                  onChange={(e) => setLayerVisibility({ ...layerVisibility, markers: e.target.checked })}
                  className="accent-accent-cyan"
                />
              </label>
            </div>
          )}
        </div>
      </div>

      {/* ============================================================ */}
      {/* BOTTOM-CENTER: DYNAMIC 12-DAY DRIFT SIMULATION PLAYER HUD    */}
      {/* ============================================================ */}
      <div className="absolute bottom-6 left-1/2 -translate-x-1/2 z-20 w-full max-w-2xl px-4 pointer-events-auto">
        <div className="bg-navy-950/95 backdrop-blur-md border border-navy-700/90 rounded-xl p-4 shadow-2xl">
          
          {/* Header Bar */}
          <div className="flex justify-between items-center mb-3">
            <div className="flex items-center space-x-3">
              <span className="w-2.5 h-2.5 rounded-full bg-accent-cyan animate-ping-slow" />
              <span className="text-white font-mono font-bold text-xs tracking-widest uppercase">
                12-DAY DRIFT HINDCAST SIMULATION
              </span>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-accent-cyan/10 border border-accent-cyan/30 text-accent-cyan font-bold">
                DAY {currentStepData.dayOffset} / 12
              </span>
            </div>

            <div className="flex items-center space-x-2">
              <span className="text-xs font-mono text-gray-400">{currentStepData.dateStr} {currentStepData.timeStr}</span>
              <button
                onClick={() => setActiveExplainer('HINDCAST')}
                className="text-gray-400 hover:text-accent-cyan p-1 transition-colors"
                title="Explain Drift Model"
              >
                <Info size={15} />
              </button>
            </div>
          </div>

          {/* Time Scrubber Slider */}
          <div className="relative flex items-center mb-3">
            <input
              type="range"
              min={0}
              max={12}
              value={simulationStep}
              onChange={(e) => {
                setSimulationStep(Number(e.target.value));
                setIsPlayingSimulation(false);
              }}
              className="w-full h-1.5 bg-navy-800 rounded-lg appearance-none cursor-pointer accent-accent-cyan"
            />
          </div>

          {/* Controls & Telemetry Readout */}
          <div className="flex items-center justify-between border-t border-navy-800/80 pt-3">
            
            {/* Playback Controls */}
            <div className="flex items-center space-x-2">
              <button
                onClick={() => {
                  setSimulationStep((prev) => Math.max(prev - 1, 0));
                  setIsPlayingSimulation(false);
                }}
                className="p-1.5 rounded-lg bg-navy-800 hover:bg-navy-700 text-gray-300 hover:text-white transition-colors"
                title="Step Back 1 Day"
              >
                <SkipBack size={14} />
              </button>

              <button
                onClick={() => setIsPlayingSimulation(!isPlayingSimulation)}
                className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-accent-cyan hover:bg-white text-navy-950 font-bold font-mono text-xs transition-colors shadow-lg"
              >
                {isPlayingSimulation ? <Pause size={14} fill="currentColor" /> : <Play size={14} fill="currentColor" />}
                <span>{isPlayingSimulation ? 'PAUSE' : 'PLAY SIMULATION'}</span>
              </button>

              <button
                onClick={() => {
                  setSimulationStep((prev) => Math.min(prev + 1, 12));
                  setIsPlayingSimulation(false);
                }}
                className="p-1.5 rounded-lg bg-navy-800 hover:bg-navy-700 text-gray-300 hover:text-white transition-colors"
                title="Step Forward 1 Day"
              >
                <SkipForward size={14} />
              </button>

              <select
                value={playbackSpeed}
                onChange={(e) => setPlaybackSpeed(Number(e.target.value))}
                className="bg-navy-800 border border-navy-700 text-gray-300 text-[11px] font-mono rounded px-2 py-1 outline-none ml-1"
              >
                <option value={1500}>0.7x Speed</option>
                <option value={1000}>1.0x Speed</option>
                <option value={500}>2.0x Speed</option>
              </select>
            </div>

            {/* Phase Description */}
            <div className="text-right">
              <span className="text-[10px] font-mono text-accent-amber block font-bold tracking-wider">
                PHASE: {currentStepData.phase}
              </span>
              <span className="text-[11px] text-gray-400 font-mono truncate max-w-xs block">
                {currentStepData.description}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* ============================================================ */}
      {/* FORENSIC EXPLAINER MODAL                                      */}
      {/* ============================================================ */}
      <ForensicExplainer
        entity={activeExplainer}
        onClose={() => setActiveExplainer(null)}
        onFlyTo={handleFlyToTarget}
      />
    </div>
  );
}
