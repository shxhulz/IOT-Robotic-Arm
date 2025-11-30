import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Bar, BarChart, ResponsiveContainer, XAxis, YAxis, Tooltip } from "recharts"

interface ObjectChartProps {
  data: { name: string; total: number }[]
}

export function ObjectChart({ data }: ObjectChartProps) {
  return (
    <Card className="glass-card border-0">
      <CardHeader>
        <CardTitle>Sorted Objects (Last 24h)</CardTitle>
      </CardHeader>
      <CardContent className="pl-2">
        <ResponsiveContainer width="100%" height={350}>
          <BarChart data={data}>
            <XAxis
              dataKey="name"
              stroke="#94a3b8"
              fontSize={12}
              tickLine={false}
              axisLine={false}
            />
            <YAxis
              stroke="#94a3b8"
              fontSize={12}
              tickLine={false}
              axisLine={false}
              tickFormatter={(value) => `${value}`}
            />
            <Tooltip 
                cursor={{fill: 'rgba(255,255,255,0.05)'}}
                contentStyle={{ 
                    backgroundColor: 'rgba(15, 23, 42, 0.9)', 
                    borderRadius: '12px', 
                    border: '1px solid rgba(255,255,255,0.1)', 
                    boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.3)',
                    color: '#fff'
                }}
            />
            <Bar 
                dataKey="total" 
                fill="currentColor" 
                radius={[6, 6, 0, 0]} 
                className="fill-primary" 
            />
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  )
}
