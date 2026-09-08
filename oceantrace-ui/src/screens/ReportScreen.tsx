import React from 'react';
import { Download, FileText, Anchor, Navigation, Target } from 'lucide-react';
import validationData from '../data/validation.json';
import wakashioData from '../data/wakashio.json';
import { motion } from 'framer-motion';

export default function ReportScreen() {
  const data = validationData;
  const overview = wakashioData;

  return (
    <div className="h-full w-full bg-navy-900 overflow-y-auto p-12 flex justify-center">
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="max-w-3xl w-full"
      >
        <div className="bg-navy-800/80 border border-navy-700 p-10 rounded-xl shadow-2xl relative overflow-hidden">
          
          {/* Header */}
          <div className="border-b border-navy-700 pb-8 mb-8">
            <h1 className="text-3xl font-light text-white tracking-widest mb-2">{overview.case_id} INVESTIGATION REPORT</h1>
            <p className="text-accent-cyan font-mono text-sm tracking-wider uppercase">Forensic Summary • {overview.date}</p>
          </div>

          <div className="space-y-8">
            {/* Observation */}
            <section>
              <h2 className="text-xs font-bold font-mono tracking-widest text-gray-500 mb-4 border-b border-navy-700 pb-2">1. OBSERVATION & DETECTION</h2>
              <div className="grid grid-cols-2 gap-6">
                <div>
                  <div className="text-sm text-gray-400 mb-1">Status</div>
                  <div className="text-white font-medium">Oil slick detected</div>
                </div>
                <div>
                  <div className="text-sm text-gray-400 mb-1">Confidence</div>
                  <div className="text-accent-cyan font-mono text-xl">87%</div>
                </div>
              </div>
            </section>

            {/* Origin */}
            <section>
              <h2 className="text-xs font-bold font-mono tracking-widest text-gray-500 mb-4 border-b border-navy-700 pb-2">2. DRIFT & ORIGIN</h2>
              <div className="grid grid-cols-2 gap-6">
                <div>
                  <div className="text-sm text-gray-400 mb-1">Status</div>
                  <div className="text-white font-medium">Estimated region backtracked</div>
                </div>
                <div>
                  <div className="text-sm text-gray-400 mb-1">Uncertainty Radius</div>
                  <div className="text-accent-amber font-mono text-xl">26.06 km</div>
                </div>
              </div>
            </section>

            {/* Vessel Analysis */}
            <section>
              <h2 className="text-xs font-bold font-mono tracking-widest text-gray-500 mb-4 border-b border-navy-700 pb-2">3. VESSEL ANALYSIS</h2>
              <div className="grid grid-cols-2 gap-6 mb-6">
                <div>
                  <div className="text-sm text-gray-400 mb-1">Candidates</div>
                  <div className="text-white font-medium">5 vessels correlated</div>
                </div>
                <div>
                  <div className="text-sm text-gray-400 mb-1">Top Candidate</div>
                  <div className="text-accent-coral font-medium text-lg">MV {data.vessel_name} (IMO {data.imo})</div>
                </div>
              </div>

              {/* Evidence Scores */}
              <div className="bg-navy-900/50 border border-navy-700 rounded p-4">
                <div className="text-sm font-bold tracking-widest text-gray-400 mb-4">EVIDENCE SCORING</div>
                
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-gray-300">Spatial Proximity</span>
                      <span className="font-mono text-accent-cyan">{data.spatial_evidence.score.toFixed(2)}</span>
                    </div>
                    <div className="w-full bg-navy-800 h-1.5 rounded-full overflow-hidden">
                      <div className="bg-accent-cyan h-full" style={{ width: `${data.spatial_evidence.score}%` }} />
                    </div>
                  </div>
                  
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-gray-300">Behavioural Deviation</span>
                      <span className="font-mono text-accent-amber">{data.behavior_evidence.score.toFixed(2)}</span>
                    </div>
                    <div className="w-full bg-navy-800 h-1.5 rounded-full overflow-hidden">
                      <div className="bg-accent-amber h-full" style={{ width: `${data.behavior_evidence.score}%` }} />
                    </div>
                  </div>

                  <div className="pt-3 mt-3 border-t border-navy-700 flex justify-between items-center">
                    <span className="text-white font-bold tracking-widest">TOTAL EVIDENCE SCORE</span>
                    <span className="text-2xl font-mono text-white">{data.total_score.toFixed(2)} / 100</span>
                  </div>
                </div>
              </div>
            </section>

            {/* Limitations */}
            <section>
              <h2 className="text-xs font-bold font-mono tracking-widest text-gray-500 mb-4 border-b border-navy-700 pb-2">4. LIMITATIONS</h2>
              <ul className="list-disc list-inside text-sm text-gray-400 space-y-2">
                <li>Single-vessel validation due to API limits.</li>
                <li>Historical reconstruction based on verified datasets.</li>
                <li>Non-calibrated score: An engineering MVP formula, not a statistical probability.</li>
                <li>Not a legal conclusion.</li>
              </ul>
            </section>
          </div>

          {/* Footer Actions */}
          <div className="mt-12 flex space-x-4 border-t border-navy-700 pt-6">
            <button className="flex items-center space-x-2 bg-navy-700 hover:bg-navy-600 text-white px-6 py-3 transition-colors text-sm font-bold tracking-widest">
              <Download size={16} />
              <span>EXPORT REPORT</span>
            </button>
            <button className="flex items-center space-x-2 bg-transparent border border-navy-600 hover:bg-navy-800 text-gray-300 px-6 py-3 transition-colors text-sm font-bold tracking-widest">
              <Download size={16} />
              <span>EXPORT MAP</span>
            </button>
          </div>

        </div>
      </motion.div>
    </div>
  );
}
