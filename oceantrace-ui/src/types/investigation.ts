export type Coordinates = [number, number];

export interface OriginData {
  origin_lon: number;
  origin_lat: number;
  origin_time: string;
  particles?: {
    lons: number[];
    lats: number[];
  };
  SYNTHETIC_TEST_DATA?: boolean;
}

export interface CandidateData {
  vessel_name: string;
  imo: string;
  planned_passage: Array<{
    pos_no: number;
    lat: string;
    lon: string;
  }>;
  actual_deviation_track: Array<{
    timestamp: string;
    lat: string;
    lon: string;
  }>;
  grounding_anchor: {
    lat: string;
    lon: string;
  };
}

export interface ValidationRecord {
  vessel_name: string;
  imo: string;
  total_score: number;
  spatial_evidence: {
    score: number;
    distance_to_origin_km: number;
    note: string;
  };
  behavior_evidence: {
    score: number;
    deviation_from_plan_km: number;
    note: string;
  };
  SYNTHETIC_TEST_DATA?: boolean;
  WARNING?: string;
}

export interface WakashioData {
  case_id: string;
  name: string;
  location: string;
  date: string;
  status: string;
  summary: string;
}

export type TimelineState = 'SATELLITE' | 'DRIFT' | 'AIS' | 'ATTRIBUTION';
