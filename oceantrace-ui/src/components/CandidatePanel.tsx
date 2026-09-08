import React from 'react';
import { Ship, ChevronRight, AlertTriangle, ShieldCheck, Crosshair } from 'lucide-react';
import clsx from 'clsx';
import { motion } from 'framer-motion';
import validationData from '../data/validation.json';
import type { TimelineState } from '../types/investigation';

interface CandidatePanelProps {
  activeState: TimelineState;
  selectedVessel: string | null;
  onSelectVessel: (v: string | null) => void;
}

export default function CandidatePanel({ activeState, selectedVessel, onSelectVessel }: CandidatePanelProps) {
  const candidates = [
    {
      id: 'v-1',
      name: validationData.vessel_name,
      imo: validationData.imo,
      score: validationData.total_score,
      status: 'TOP MATCH (#1)',
      flag: 'Panama (Bulk Carrier)',
      spatialScore: '98.76% (0.62 km to origin)',
      behaviorScore: '86.64% (43.32 km deviation)',
      borderColor: 'border-accent-coral',
      textColor: 'text-accent-coral',
      bgColor: 'bg-accent-coral/10 hover:bg-accent-coral/20'
    },
    {
      id: 'v-2',
      name: 'MSC Alessia',
      imo: '9225249',
      score: 48.3,
      status: 'CLEARED (SAFE PASSAGE)',
      flag: 'Liberia (Container)',
      spatialScore: '42.1% (38.4 km to origin)',
      behaviorScore: '94.2% (Maintained corridor)',
      borderColor: 'border-navy-700',
      textColor: 'text-gray-400',
      bgColor: 'bg-navy-900/60 hover:bg-navy-800'
    },
    {
      id: 'v-3',
      name: 'Pacific Crown',
      imo: '9438810',
      score: 31.8,
      status: 'CLEARED (TIMING MISMATCH)',
      flag: 'Singapore (Tanker)',
      spatialScore: '18.4% (84.1 km to origin)',
      behaviorScore: '89.0% (Maintained corridor)',
      borderColor: 'border-navy-700',
      textColor: 'text-gray-400',
      bgColor: 'bg-navy-900/60 hover:bg-navy-800'
    }
  ];

  return (
    <div className="flex flex-col space-y-3 font-mono">
      {candidates.map((c, i) => {
        const isSelected = selectedVessel === c.id;
        const isTop = c.id === 'v-1';

        return (
          <motion.div
            key={c.id}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.08 }}
            onClick={() => onSelectVessel(isSelected ? null : c.id)}
            className={clsx(
              "cursor-pointer border rounded-lg p-3 transition-all duration-200 group text-left",
              isSelected 
                ? "border-accent-cyan bg-navy-800/90 shadow-lg shadow-accent-cyan/10" 
                : isTop 
                  ? "border-accent-coral/60 bg-accent-coral/10 hover:bg-accent-coral/20" 
                  : "border-navy-800 bg-navy-950/60 hover:bg-navy-800/40"
            )}
          >
            {/* Header */}
            <div className="flex justify-between items-start mb-1.5">
              <div className="flex items-center space-x-2">
                <Ship size={14} className={isTop ? "text-accent-coral" : "text-gray-500"} />
                <span className={clsx("text-xs font-bold font-sans", isTop ? "text-white" : "text-gray-300")}>
                  {c.name}
                </span>
              </div>
              <span className={clsx("text-sm font-bold", isTop ? "text-accent-coral" : "text-gray-500")}>
                {c.score}%
              </span>
            </div>

            {/* Subtitle / Flag */}
            <div className="flex justify-between items-center text-[10px] text-gray-500 mb-2">
              <span>IMO {c.imo} • {c.flag}</span>
              <span className={clsx("text-[9px] px-1.5 py-0.5 rounded uppercase font-bold", isTop ? "bg-accent-coral/20 text-accent-coral" : "bg-navy-800 text-gray-400")}>
                {c.status}
              </span>
            </div>

            {/* Quick Metrics */}
            <div className="space-y-1 text-[10px] bg-navy-950/80 p-2 rounded border border-navy-800/80 text-gray-400">
              <div className="flex justify-between">
                <span>Spatial Match:</span>
                <span className="text-gray-300">{c.spatialScore}</span>
              </div>
              <div className="flex justify-between">
                <span>Route Deviation:</span>
                <span className="text-gray-300">{c.behaviorScore}</span>
              </div>
            </div>

            {/* Click CTA */}
            <div className="mt-2 flex items-center justify-between text-[10px] text-accent-cyan opacity-80 group-hover:opacity-100 transition-opacity">
              <span>{isSelected ? "CLOSE DOSSIER" : "CLICK FOR FORENSIC DOSSIER"}</span>
              <ChevronRight size={12} className={clsx("transform transition-transform", isSelected ? "rotate-90" : "group-hover:translate-x-1")} />
            </div>
          </motion.div>
        );
      })}
    </div>
  );
}
