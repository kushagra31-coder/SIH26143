import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Target, Navigation, Clock, ShieldAlert } from 'lucide-react';
import validationData from '../data/validation.json';

export default function EvidenceDrawer({ isOpen, vesselId, onClose }: { isOpen: boolean, vesselId: string | null, onClose: () => void }) {
  if (vesselId !== 'v-1') return null; // We only have real data for Wakashio (v-1) in this MVP

  const data = validationData;

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ x: '100%', opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          exit={{ x: '100%', opacity: 0 }}
          transition={{ type: 'spring', damping: 25, stiffness: 200 }}
          className="absolute right-0 top-0 bottom-0 w-[400px] bg-navy-900 border-l border-navy-700 z-30 shadow-2xl overflow-y-auto"
        >
          {/* Header */}
          <div className="flex justify-between items-start p-6 border-b border-navy-800 bg-navy-800/50">
            <div>
              <h2 className="text-xs font-mono text-gray-500 tracking-widest mb-1">WHY RANKED #1</h2>
              <h1 className="text-xl font-bold text-white tracking-wide">MV {data.vessel_name.toUpperCase()}</h1>
              <div className="text-sm font-mono text-gray-400">IMO {data.imo}</div>
            </div>
            <button onClick={onClose} className="p-2 hover:bg-navy-700 rounded-full text-gray-400 hover:text-white transition-colors">
              <X size={20} />
            </button>
          </div>

          <div className="p-6 space-y-8">
            
            {/* Warning Banner if Synthetic */}
            {data.SYNTHETIC_TEST_DATA && (
              <div className="bg-red-900/20 border border-red-500/50 p-4 rounded flex items-start space-x-3">
                <ShieldAlert className="text-red-500 flex-shrink-0 mt-0.5" size={16} />
                <div className="text-xs text-red-200/80 font-mono leading-relaxed">
                  {data.WARNING}
                </div>
              </div>
            )}

            {/* Spatial Evidence */}
            <section>
              <div className="flex items-center space-x-2 mb-3">
                <Target size={16} className="text-accent-cyan" />
                <h3 className="text-sm font-bold tracking-widest text-gray-300">SPATIAL EVIDENCE</h3>
              </div>
              <div className="bg-navy-800 p-4 rounded border border-navy-700">
                <div className="flex justify-between items-end mb-2">
                  <span className="text-3xl font-mono text-white">{data.spatial_evidence.score.toFixed(2)}</span>
                  <span className="text-xs text-gray-500 mb-1">/ 100</span>
                </div>
                <div className="w-full bg-navy-900 h-2 rounded-full overflow-hidden mb-3">
                  <div className="bg-accent-cyan h-full" style={{ width: `${data.spatial_evidence.score}%` }} />
                </div>
                <p className="text-sm text-gray-400">
                  <span className="text-white font-medium">{data.spatial_evidence.distance_to_origin_km.toFixed(2)} km</span> from inferred origin.
                </p>
                <p className="text-xs text-gray-500 mt-2 font-mono">{data.spatial_evidence.note}</p>
              </div>
            </section>

            {/* Behavioural Evidence */}
            <section>
              <div className="flex items-center space-x-2 mb-3">
                <Navigation size={16} className="text-accent-amber" />
                <h3 className="text-sm font-bold tracking-widest text-gray-300">BEHAVIOURAL EVIDENCE</h3>
              </div>
              <div className="bg-navy-800 p-4 rounded border border-navy-700">
                <div className="flex justify-between items-end mb-2">
                  <span className="text-3xl font-mono text-white">{data.behavior_evidence.score.toFixed(2)}</span>
                  <span className="text-xs text-gray-500 mb-1">/ 100</span>
                </div>
                <div className="w-full bg-navy-900 h-2 rounded-full overflow-hidden mb-3">
                  <div className="bg-accent-amber h-full" style={{ width: `${data.behavior_evidence.score}%` }} />
                </div>
                <p className="text-sm text-gray-400">
                  <span className="text-white font-medium">{data.behavior_evidence.deviation_from_plan_km.toFixed(2)} km</span> route deviation detected.
                </p>
                <p className="text-xs text-gray-500 mt-2 font-mono">{data.behavior_evidence.note}</p>
              </div>
            </section>

            {/* Conclusion */}
            <section className="border-t border-navy-700 pt-6">
              <div className="bg-accent-coral/10 border border-accent-coral/30 p-4 text-center rounded">
                <div className="text-sm font-bold text-accent-coral tracking-widest mb-1">STRONG INVESTIGATION LEAD</div>
                <div className="text-xs font-mono text-gray-500">Not a legal conclusion.</div>
              </div>
            </section>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
