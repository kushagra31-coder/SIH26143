import React from 'react';
import { Play, SkipBack, FastForward } from 'lucide-react';
import clsx from 'clsx';
import type { TimelineState } from '../types/investigation';

const STAGES: TimelineState[] = ['SATELLITE', 'DRIFT', 'AIS', 'ATTRIBUTION'];

export default function TimelineControls({ activeState, onStateChange }: { activeState: TimelineState, onStateChange: (s: TimelineState) => void }) {
  
  const currentIndex = STAGES.indexOf(activeState);

  const handlePlay = () => {
    if (currentIndex < STAGES.length - 1) {
      onStateChange(STAGES[currentIndex + 1]);
    } else {
      onStateChange(STAGES[0]); // loop back
    }
  };

  const handleBacktrack = () => {
    if (currentIndex > 0) {
      onStateChange(STAGES[currentIndex - 1]);
    }
  };

  const handleForward = () => {
    if (currentIndex < STAGES.length - 1) {
      onStateChange(STAGES[currentIndex + 1]);
    }
  };

  return (
    <div className="bg-navy-900/90 backdrop-blur-md border border-navy-700 p-4 rounded-xl shadow-2xl flex flex-col items-center">
      <div className="flex items-center justify-between w-full mb-4 px-8 relative">
        {/* Timeline Bar */}
        <div className="absolute left-10 right-10 h-0.5 bg-navy-700 top-1/2 -translate-y-1/2 z-0" />
        <div 
          className="absolute left-10 h-0.5 bg-accent-cyan top-1/2 -translate-y-1/2 z-0 transition-all duration-500"
          style={{ width: `calc(${currentIndex * 33.33}% - 20px)` }}
        />

        {STAGES.map((s, idx) => {
          const isActive = idx <= currentIndex;
          return (
            <div key={s} className="relative z-10 flex flex-col items-center cursor-pointer" onClick={() => onStateChange(s)}>
              <div className={clsx(
                "w-3 h-3 rounded-full transition-colors duration-300",
                isActive ? "bg-accent-cyan shadow-[0_0_10px_rgba(34,211,238,0.6)]" : "bg-navy-600"
              )} />
              <span className={clsx(
                "absolute top-5 text-[10px] font-mono tracking-widest whitespace-nowrap",
                isActive ? "text-gray-300" : "text-gray-600"
              )}>
                {s}
              </span>
            </div>
          );
        })}
      </div>

      <div className="flex items-center space-x-6 mt-6">
        <button 
          onClick={handleBacktrack}
          className="flex items-center space-x-2 text-gray-400 hover:text-white transition-colors"
        >
          <SkipBack size={16} />
          <span className="text-xs font-bold tracking-widest">BACKTRACK</span>
        </button>

        <button 
          onClick={handlePlay}
          className="bg-accent-cyan text-navy-900 p-3 rounded-full hover:bg-white transition-colors shadow-[0_0_15px_rgba(34,211,238,0.4)]"
        >
          <Play size={20} fill="currentColor" className="ml-1" />
        </button>

        <button 
          onClick={handleForward}
          className="flex items-center space-x-2 text-gray-400 hover:text-white transition-colors"
        >
          <span className="text-xs font-bold tracking-widest">FORWARD DRIFT</span>
          <FastForward size={16} />
        </button>
      </div>
    </div>
  );
}
