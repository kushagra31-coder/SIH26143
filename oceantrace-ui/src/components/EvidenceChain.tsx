import React from 'react';
import { CheckCircle2, Circle } from 'lucide-react';
import clsx from 'clsx';
import type { TimelineState } from '../types/investigation';

const STAGES: { id: TimelineState; label: string; desc: string }[] = [
  { id: 'SATELLITE', label: 'SATELLITE', desc: 'Slick detected' },
  { id: 'DRIFT', label: 'DRIFT', desc: 'Origin estimated' },
  { id: 'AIS', label: 'AIS', desc: 'Vessels correlated' },
  { id: 'ATTRIBUTION', label: 'ATTRIBUTION', desc: 'Candidate ranked' }
];

export default function EvidenceChain({ activeState, onStateSelect }: { activeState: TimelineState, onStateSelect: (s: TimelineState) => void }) {
  const activeIdx = STAGES.findIndex(s => s.id === activeState);

  return (
    <div className="relative">
      <div className="absolute left-2.5 top-3 bottom-4 w-px bg-navy-700" />
      
      <div className="flex flex-col space-y-6">
        {STAGES.map((stage, idx) => {
          const isActive = idx === activeIdx;
          const isPast = idx < activeIdx;
          const isFuture = idx > activeIdx;
          
          return (
            <button 
              key={stage.id} 
              onClick={() => onStateSelect(stage.id)}
              className={clsx(
                "relative flex items-start text-left group",
                isFuture ? "opacity-40" : "opacity-100"
              )}
            >
              <div className="bg-navy-900 py-1 z-10 mr-4">
                {(isActive || isPast) ? (
                  <CheckCircle2 size={20} className={isActive ? "text-accent-cyan" : "text-gray-500"} />
                ) : (
                  <Circle size={20} className="text-navy-600" />
                )}
              </div>
              <div className="pt-0.5">
                <div className={clsx("text-sm font-bold tracking-widest transition-colors", isActive ? "text-white" : "text-gray-400 group-hover:text-gray-300")}>
                  {stage.label}
                </div>
                <div className="text-xs font-mono text-gray-500 mt-1">
                  {stage.desc}
                </div>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
