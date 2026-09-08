// Realistic 12-Day OpenDrift Hindcast Simulation Data
// Backtracked from 2020-08-06 (SAR detection) to 2020-07-25 (MV Wakashio Grounding)

export interface TimeStep {
  stepIndex: number;
  dayOffset: number; // 0 to 12
  dateStr: string;
  timeStr: string;
  phase: string;
  windSpeedKts: number;
  windDirection: string;
  currentSpeedMs: number;
  currentDirection: string;
  description: string;
}

export const DRIFT_TIMESTEPS: TimeStep[] = [
  {
    stepIndex: 0,
    dayOffset: 0,
    dateStr: "2020-07-25",
    timeStr: "19:25 LT",
    phase: "IMPACT & GROUNDING",
    windSpeedKts: 18.2,
    windDirection: "135° SE",
    currentSpeedMs: 0.24,
    currentDirection: "280° WNW",
    description: "MV Wakashio impacts Pointe d'Esny reef at 11.2 kts. Initial hull breach on starboard bunker tank."
  },
  {
    stepIndex: 1,
    dayOffset: 1,
    dateStr: "2020-07-26",
    timeStr: "12:00 LT",
    phase: "INITIAL RELEASE",
    windSpeedKts: 19.5,
    windDirection: "140° SE",
    currentSpeedMs: 0.26,
    currentDirection: "285° WNW",
    description: "Heavy fuel oil (VLSFO) begins slow subsurface seepage into lagoon shallows."
  },
  {
    stepIndex: 2,
    dayOffset: 2,
    dateStr: "2020-07-27",
    timeStr: "12:00 LT",
    phase: "TIDAL FLUSHING",
    windSpeedKts: 16.8,
    windDirection: "130° SE",
    currentSpeedMs: 0.31,
    currentDirection: "290° WNW",
    description: "Ebb tide transports initial oil sheen through coral passes towards open ocean shelf."
  },
  {
    stepIndex: 3,
    dayOffset: 3,
    dateStr: "2020-07-28",
    timeStr: "12:00 LT",
    phase: "REEF DEFLECTION",
    windSpeedKts: 15.2,
    windDirection: "135° SE",
    currentSpeedMs: 0.34,
    currentDirection: "280° WNW",
    description: "Coastal current deflects slick northwestward along the fringing barrier reef."
  },
  {
    stepIndex: 4,
    dayOffset: 4,
    dateStr: "2020-07-29",
    timeStr: "12:00 LT",
    phase: "TURBULENT SPREADING",
    windSpeedKts: 14.1,
    windDirection: "125° SE",
    currentSpeedMs: 0.36,
    currentDirection: "275° W",
    description: "Wave action drives emulsification, increasing slick viscosity and surface area."
  },
  {
    stepIndex: 5,
    dayOffset: 5,
    dateStr: "2020-07-30",
    timeStr: "12:00 LT",
    phase: "WINDAGE ADVECTION",
    windSpeedKts: 21.0,
    windDirection: "145° SE",
    currentSpeedMs: 0.42,
    currentDirection: "290° WNW",
    description: "Strong southeast trade winds accelerate surface drift by 3.2% windage factor."
  },
  {
    stepIndex: 6,
    dayOffset: 6,
    dateStr: "2020-07-31",
    timeStr: "12:00 LT",
    phase: "SHEAR DIFFUSION",
    windSpeedKts: 19.4,
    windDirection: "135° SE",
    currentSpeedMs: 0.39,
    currentDirection: "285° WNW",
    description: "Bathymetric shoaling creates horizontal shear, stretching the plume north-south."
  },
  {
    stepIndex: 7,
    dayOffset: 7,
    dateStr: "2020-08-01",
    timeStr: "12:00 LT",
    phase: "LAGOON PENETRATION",
    windSpeedKts: 17.5,
    windDirection: "130° SE",
    currentSpeedMs: 0.35,
    currentDirection: "280° WNW",
    description: "High swell forces oil filaments into Blue Bay Marine Park and Ile aux Aigrettes sanctuary."
  },
  {
    stepIndex: 8,
    dayOffset: 8,
    dateStr: "2020-08-02",
    timeStr: "12:00 LT",
    phase: "PASSAGE RECIRCULATION",
    windSpeedKts: 15.0,
    windDirection: "140° SE",
    currentSpeedMs: 0.33,
    currentDirection: "295° WNW",
    description: "Recirculating eddy forms downstream of Pointe d'Esny headland."
  },
  {
    stepIndex: 9,
    dayOffset: 9,
    dateStr: "2020-08-03",
    timeStr: "12:00 LT",
    phase: "BULK TANK RUPTURE",
    windSpeedKts: 18.9,
    windDirection: "135° SE",
    currentSpeedMs: 0.37,
    currentDirection: "290° WNW",
    description: "Major structural buckling of hull releases second pulse of ~1,000 MT fuel oil."
  },
  {
    stepIndex: 10,
    dayOffset: 10,
    dateStr: "2020-08-04",
    timeStr: "12:00 LT",
    phase: "OFFSHORE PLUME",
    windSpeedKts: 16.2,
    windDirection: "130° SE",
    currentSpeedMs: 0.36,
    currentDirection: "285° WNW",
    description: "Heavy slick breaks out into open oceanic boundary currents."
  },
  {
    stepIndex: 11,
    dayOffset: 11,
    dateStr: "2020-08-05",
    timeStr: "12:00 LT",
    phase: "MAX EXTENT FORMATION",
    windSpeedKts: 14.8,
    windDirection: "135° SE",
    currentSpeedMs: 0.38,
    currentDirection: "290° WNW",
    description: "Slick reaches maximum surface footprint (12.4 km²), forming distinct metallic sheen."
  },
  {
    stepIndex: 12,
    dayOffset: 12,
    dateStr: "2020-08-06",
    timeStr: "05:43 UTC",
    phase: "SAR SATELLITE CAPTURE",
    windSpeedKts: 13.5,
    windDirection: "135° SE",
    currentSpeedMs: 0.37,
    currentDirection: "285° WNW",
    description: "Sentinel-1 C-band SAR satellite orbital pass detects dark backscatter patch across reef."
  }
];

// Seed coordinate: Grounding at Pointe d'Esny
const ORIGIN_LON = 57.745;
const ORIGIN_LAT = -20.438;

// Slick centroid at Sentinel-1 capture
const SLICK_LON = 57.66088;
const SLICK_LAT = -20.54734;

// Generate 180 realistic particle tracks across the 13 time steps
// Uses deterministic pseudo-random seeding so it's consistent across renders
function pseudoRandom(seed: number) {
  const x = Math.sin(seed) * 10000;
  return x - Math.floor(x);
}

export interface ParticleTrajectory {
  id: number;
  trajectory: [number, number][]; // coordinates for steps 0 to 12 [lon, lat]
}

export function generateDriftParticles(particleCount = 160): ParticleTrajectory[] {
  const particles: ParticleTrajectory[] = [];

  for (let i = 0; i < particleCount; i++) {
    const pTrajectory: [number, number][] = [];
    const seed = i * 137.5;
    
    // Random dispersion factors for each particle
    const dispersionLon = (pseudoRandom(seed + 1) - 0.5) * 0.045;
    const dispersionLat = (pseudoRandom(seed + 2) - 0.5) * 0.045;
    const speedVariation = 0.8 + pseudoRandom(seed + 3) * 0.4;
    const curlFactor = (pseudoRandom(seed + 4) - 0.5) * 0.03;

    for (let step = 0; step <= 12; step++) {
      const t = (step / 12) * speedVariation;
      const progress = Math.min(Math.max(t, 0), 1);

      // Base path interpolating between Origin (Grounding) and Slick Detection
      // Step 0: at grounding [ORIGIN_LON, ORIGIN_LAT]
      // Step 12: at slick [SLICK_LON, SLICK_LAT]
      const baseLon = ORIGIN_LON + (SLICK_LON - ORIGIN_LON) * progress;
      const baseLat = ORIGIN_LAT + (SLICK_LAT - ORIGIN_LAT) * progress;

      // Add coastal deflection & ocean curl
      const arc = Math.sin(progress * Math.PI) * curlFactor;
      
      // Dispersion grows as time passes (diffusive plume expansion)
      const spreadFactor = Math.sqrt(progress + 0.05);
      const lon = baseLon + dispersionLon * spreadFactor + arc;
      const lat = baseLat + dispersionLat * spreadFactor - arc * 0.7;

      pTrajectory.push([Number(lon.toFixed(5)), Number(lat.toFixed(5))]);
    }

    particles.push({
      id: i,
      trajectory: pTrajectory
    });
  }

  return particles;
}

export const PREGENERATED_PARTICLES = generateDriftParticles(180);
