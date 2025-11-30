import { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

interface HistoryData {
  time: string;
  Plastic: number;
  Metal: number;
  Paper: number;
}

export function LineChartInteractive() {
  const [data, setData] = useState<HistoryData[]>([]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch('http://localhost:8000/stats/history');
        const jsonData = await response.json();
        setData(jsonData);
      } catch (error) {
        console.error("Error fetching history:", error);
      }
    };

    fetchData();
    // Refresh every 30 seconds
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, []);

  return (
    <Card className="glass-card border-0 col-span-4">
      <CardHeader>
        <CardTitle className="text-xl font-bold font-mono">Material Sorting Trends</CardTitle>
        <CardDescription className="text-zinc-400">
          Hourly breakdown of sorted materials (Last 24h)
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="h-[300px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" vertical={false} />
              <XAxis 
                dataKey="time" 
                stroke="#888888" 
                fontSize={12} 
                tickLine={false} 
                axisLine={false}
                tickFormatter={(value) => {
                  const date = new Date(value);
                  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
                }}
              />
              <YAxis 
                stroke="#888888" 
                fontSize={12} 
                tickLine={false} 
                axisLine={false}
                tickFormatter={(value) => `${value}`}
              />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: 'rgba(0,0,0,0.8)', 
                  border: '1px solid rgba(255,255,255,0.1)',
                  borderRadius: '12px',
                  boxShadow: '0 4px 20px rgba(0,0,0,0.5)'
                }}
                itemStyle={{ color: '#fff' }}
                cursor={{ stroke: 'rgba(255,255,255,0.2)', strokeWidth: 2 }}
                labelFormatter={(label) => new Date(label).toLocaleString()}
              />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="Plastic" 
                stroke="#06b6d4" // Cyan
                strokeWidth={3}
                dot={false}
                activeDot={{ r: 6, strokeWidth: 0 }}
                animationDuration={1500}
              />
              <Line 
                type="monotone" 
                dataKey="Metal" 
                stroke="#8b5cf6" // Violet
                strokeWidth={3}
                dot={false}
                activeDot={{ r: 6, strokeWidth: 0 }}
                animationDuration={1500}
              />
              <Line 
                type="monotone" 
                dataKey="Paper" 
                stroke="#f59e0b" // Amber
                strokeWidth={3}
                dot={false}
                activeDot={{ r: 6, strokeWidth: 0 }}
                animationDuration={1500}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}
