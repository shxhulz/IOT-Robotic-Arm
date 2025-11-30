import { useEffect, useState } from 'react'
import { Activity, Box, CheckCircle, AlertTriangle, Wifi, WifiOff } from 'lucide-react'
import { StatsCard } from '@/components/StatsCard'
import { ActivityLog } from '@/components/ActivityLog'
import { ObjectChart } from '@/components/ObjectChart'
import { LineChartInteractive } from '@/components/LineChartInteractive'
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

interface LogEntry {
  id: number
  timestamp: string
  message: string
  type: string
}

function App() {
  const [isConnected, setIsConnected] = useState(false)
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [stats, setStats] = useState({
    total: 0,
    paper: 0,
    metal: 0,
    plastic: 0,
    successRate: 100
  })
  const [telemetry, setTelemetry] = useState({ distance: 0 })

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws')

    ws.onopen = () => {
      setIsConnected(true)
      addLog('System connected', 'info')
    }

    ws.onclose = () => {
      setIsConnected(false)
      addLog('System disconnected', 'error')
    }

    ws.onmessage = (event) => {
      const message = JSON.parse(event.data)
      const { topic, data } = message

      if (topic === 'robot_events') {
        handleEvent(data)
      } else if (topic === 'robot_telemetry') {
        setTelemetry({ distance: data.distance })
      }
    }

    return () => {
      ws.close()
    }
  }, [])

  useEffect(() => {
    fetch('http://localhost:8000/stats/today')
      .then(res => res.json())
      .then(data => {
        if (data.class_breakdown) {
            setStats(prev => ({
                ...prev,
                total: data.total_events_24h,
                paper: data.class_breakdown.paper || 0,
                metal: data.class_breakdown.metal || 0,
                plastic: data.class_breakdown.plastic || 0
            }))
        }
      })
      .catch(err => console.error("Failed to fetch stats", err))
  }, [])

  const addLog = (message: string, type: string = 'info') => {
    setLogs(prev => [...prev.slice(-49), {
      id: Date.now(),
      timestamp: new Date().toISOString(),
      message,
      type
    }])
  }

  const handleEvent = (data: any) => {
    const { event_type, object_class } = data
    
    if (event_type === 'detection') {
      addLog(`Detected ${object_class}`, 'info')
    } else if (event_type === 'pickup_success') {
      addLog(`Successfully sorted ${object_class}`, 'success')
      setStats(prev => ({
        ...prev,
        total: prev.total + 1,
        [object_class]: (prev[object_class as keyof typeof prev] as number) + 1
      }))
    } else if (event_type === 'pickup_fail') {
      addLog(`Failed to sort object`, 'error')
    }
  }

  const chartData = [
    { name: 'Paper', total: stats.paper },
    { name: 'Metal', total: stats.metal },
    { name: 'Plastic', total: stats.plastic },
  ]

  return (
    <div className="min-h-screen p-8 text-foreground selection:bg-white selection:text-black">
      <div className="max-w-7xl mx-auto space-y-12">
        {/* Header Section */}
        <div className="flex flex-col items-center justify-center space-y-6 py-12 animate-fade-in">
          <div className="relative">
             <div className="absolute -inset-4 rounded-full bg-white/10 blur-2xl opacity-50 animate-pulse-slow"></div>
             <div className="relative bg-gradient-to-br from-zinc-800 to-black p-4 rounded-2xl border border-white/10 shadow-xl animate-float hover:scale-110 transition-transform duration-500 cursor-pointer group">
                <Box className="h-12 w-12 text-white group-hover:rotate-12 transition-transform duration-500" strokeWidth={1.5} />
             </div>
          </div>
          <div className="text-center space-y-4">
            <h1 className="text-5xl md:text-7xl font-bold tracking-tighter text-gradient glow-text">
              Robotic Arm<br />Analytics
            </h1>
            <p className="text-xl text-zinc-400 max-w-2xl mx-auto font-light">
              Beyond monitoring. Beyond control.<br />
              The next layer of intelligence where humans and machines work together.
            </p>
          </div>
          
          <div className="flex items-center space-x-3 bg-zinc-900/80 px-6 py-3 rounded-full border border-white/10 backdrop-blur-md mt-8 shadow-lg hover:bg-zinc-800/80 transition-colors cursor-default">
            {isConnected ? <div className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse shadow-[0_0_10px_#34d399]" /> : <div className="w-2 h-2 bg-red-500 rounded-full" />}
            <span className="font-mono text-sm tracking-widest uppercase text-zinc-300">
              {isConnected ? "System Online" : "System Offline"}
            </span>
          </div>
        </div>

        {/* Stats Grid */}
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4 animate-slide-up">
          <StatsCard title="Total Sorted" value={stats.total} icon={Box} />
          <StatsCard title="Success Rate" value={`${stats.successRate}%`} icon={CheckCircle} />
          <StatsCard title="Distance Sensor" value={`${telemetry.distance}cm`} icon={Activity} />
          <StatsCard title="Active Robot" value="Robot 1" icon={AlertTriangle} description="ID: robot_1" />
        </div>

        {/* Charts & Logs */}
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-7 animate-slide-up" style={{ animationDelay: '0.1s' }}>
          <div className="col-span-4">
            <ObjectChart data={chartData} />
          </div>
          <div className="col-span-3">
            <ActivityLog logs={logs} />
          </div>
        </div>
        
        {/* Charts Section */}
        <div className="grid gap-6 md:grid-cols-7 animate-slide-up" style={{ animationDelay: "0.2s" }}>
          <div className="col-span-7">
             <LineChartInteractive />
          </div>
        </div>
        
        {/* Status Footer */}
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-7 animate-slide-up" style={{ animationDelay: '0.2s' }}>
           <div className="col-span-7">
             <Card className="glass-card border-0 rounded-3xl bg-zinc-900/20">
                <CardContent className="py-6 px-8">
                    <div className="flex justify-between items-center text-sm text-zinc-500 font-mono">
                        <div className="flex space-x-8">
                            <div className="flex items-center space-x-3">
                                <div className={`w-1.5 h-1.5 rounded-full ${isConnected ? 'bg-emerald-400 shadow-[0_0_8px_#34d399]' : 'bg-red-500'}`} />
                                <span>BACKEND_CONNECTION</span>
                            </div>
                            <div className="flex items-center space-x-3">
                                <div className={`w-1.5 h-1.5 rounded-full ${telemetry.distance > 0 ? 'bg-blue-400 shadow-[0_0_8px_#60a5fa]' : 'bg-zinc-700'}`} />
                                <span>TELEMETRY_STREAM</span>
                            </div>
                        </div>
                        <div>v2.0.0</div>
                    </div>
                </CardContent>
             </Card>
           </div>
        </div>
      </div>
    </div>
  )
}

export default App
