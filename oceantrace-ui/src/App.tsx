import { useState } from 'react';
import OverviewScreen from './screens/OverviewScreen';
import InvestigationScreen from './screens/InvestigationScreen';
import ReportScreen from './screens/ReportScreen';
import { Layers, Activity, FileText, Globe, ChevronDown, Bell, Settings, HelpCircle } from 'lucide-react';
import clsx from 'clsx';

type Screen = 'OVERVIEW' | 'INVESTIGATION' | 'REPORT';

function App() {
  const [activeScreen, setActiveScreen] = useState<Screen>('OVERVIEW');

  return (
    <div className="flex flex-col h-screen w-screen overflow-hidden" style={{ background: 'var(--color-cmems-950)', color: 'var(--color-text-primary)', fontFamily: 'var(--font-sans)' }}>
      
      {/* ================================================
          TOP NAVIGATION BAR — Copernicus Marine Style
          ================================================ */}
      <header className="flex-none flex items-center justify-between h-[52px] px-4 border-b" style={{ background: 'rgba(5, 20, 38, 0.98)', borderColor: 'rgba(59, 130, 246, 0.18)', backdropFilter: 'blur(12px)', zIndex: 50 }}>
        
        {/* Left: Brand */}
        <div className="flex items-center space-x-4">
          {/* Logo Mark */}
          <div className="flex items-center space-x-2.5">
            <div className="relative w-7 h-7">
              <div className="absolute inset-0 rounded-full border-2" style={{ borderColor: 'var(--color-ocean-cyan)', opacity: 0.8 }} />
              <div className="absolute inset-1 rounded-full" style={{ background: 'linear-gradient(135deg, var(--color-ocean-cyan) 0%, var(--color-ocean-teal) 100%)', opacity: 0.6 }} />
              <Globe className="absolute inset-0 m-auto w-3.5 h-3.5" style={{ color: 'var(--color-text-primary)' }} />
            </div>
            <div>
              <div className="text-sm font-bold tracking-widest uppercase" style={{ color: 'var(--color-text-primary)', letterSpacing: '0.14em' }}>OceanTrace</div>
              <div className="text-[9px] font-medium tracking-widest uppercase" style={{ color: 'var(--color-ocean-cyan)', fontFamily: 'var(--font-mono)', letterSpacing: '0.16em' }}>Maritime Intelligence Console</div>
            </div>
          </div>

          {/* Divider */}
          <div className="w-px h-7" style={{ background: 'rgba(59, 130, 246, 0.2)' }} />

          {/* Nav Tabs */}
          <nav className="flex items-center space-x-1">
            {([
              { id: 'OVERVIEW', label: 'Case Overview', icon: Layers },
              { id: 'INVESTIGATION', label: 'Investigation', icon: Activity },
              { id: 'REPORT', label: 'Evidence Report', icon: FileText },
            ] as { id: Screen; label: string; icon: any }[]).map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                onClick={() => setActiveScreen(id)}
                className="flex items-center space-x-1.5 px-3.5 py-2 rounded-md text-xs font-medium transition-all duration-150"
                style={{
                  color: activeScreen === id ? 'var(--color-text-primary)' : 'var(--color-text-secondary)',
                  background: activeScreen === id ? 'rgba(59, 130, 246, 0.15)' : 'transparent',
                  borderBottom: activeScreen === id ? '2px solid var(--color-ocean-cyan)' : '2px solid transparent',
                  fontFamily: 'var(--font-sans)',
                  borderRadius: activeScreen === id ? '6px 6px 0 0' : '6px',
                }}
              >
                <Icon size={13} style={{ color: activeScreen === id ? 'var(--color-ocean-cyan)' : 'inherit' }} />
                <span>{label}</span>
              </button>
            ))}
          </nav>
        </div>

        {/* Right: Status & Controls */}
        <div className="flex items-center space-x-3">
          {/* Data Status Badge */}
          <div className="flex items-center space-x-2 px-3 py-1.5 rounded-md" style={{ background: 'rgba(5, 150, 105, 0.12)', border: '1px solid rgba(5, 150, 105, 0.3)' }}>
            <span className="w-1.5 h-1.5 rounded-full" style={{ background: 'var(--color-ocean-green)' }}>
              <span className="block w-full h-full rounded-full animate-radar-ring" style={{ background: 'var(--color-ocean-green)' }} />
            </span>
            <span className="text-[10px] font-semibold tracking-wider uppercase" style={{ fontFamily: 'var(--font-mono)', color: 'var(--color-ocean-green)' }}>LIVE DATA</span>
          </div>

          {/* Product Selector */}
          <div className="flex items-center space-x-1.5 px-3 py-1.5 rounded-md cursor-pointer transition-all" style={{ background: 'rgba(10, 38, 68, 0.8)', border: '1px solid rgba(59, 130, 246, 0.2)' }}>
            <span className="text-[11px]" style={{ color: 'var(--color-text-secondary)', fontFamily: 'var(--font-mono)' }}>MFS-WAKASHIO-2020</span>
            <ChevronDown size={11} style={{ color: 'var(--color-text-muted)' }} />
          </div>

          {/* Divider */}
          <div className="w-px h-5" style={{ background: 'rgba(59, 130, 246, 0.15)' }} />

          {/* Icon Buttons */}
          {[Bell, Settings, HelpCircle].map((Icon, i) => (
            <button key={i} className="p-1.5 rounded-md transition-colors" style={{ color: 'var(--color-text-muted)' }}
              onMouseEnter={e => (e.currentTarget.style.color = 'var(--color-text-primary)')}
              onMouseLeave={e => (e.currentTarget.style.color = 'var(--color-text-muted)')}>
              <Icon size={15} />
            </button>
          ))}

          {/* User Avatar */}
          <div className="w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold" style={{ background: 'linear-gradient(135deg, #2361aa, #0d9488)', color: 'white' }}>
            OT
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 relative overflow-hidden">
        {activeScreen === 'OVERVIEW' && <OverviewScreen onStart={() => setActiveScreen('INVESTIGATION')} />}
        {activeScreen === 'INVESTIGATION' && <InvestigationScreen />}
        {activeScreen === 'REPORT' && <ReportScreen />}
      </main>
    </div>
  );
}

export default App;
