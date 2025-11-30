import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ScrollArea } from "@/components/ui/scroll-area" // We need to create this or use div
import { useEffect, useRef } from "react"

interface LogEntry {
  id: number
  timestamp: string
  message: string
  type: string
}

interface ActivityLogProps {
  logs: LogEntry[]
}

export function ActivityLog({ logs }: ActivityLogProps) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [logs])

  return (
    <Card className="h-[400px] flex flex-col glass-card border-0">
      <CardHeader>
        <CardTitle>Activity Log</CardTitle>
      </CardHeader>
      <CardContent className="flex-1 overflow-hidden p-0">
        <ScrollArea className="h-full">
            <div className="p-4 space-y-2">
            {logs.map((log, index) => (
                <div key={index} className="text-sm border-b border-white/10 pb-2 last:border-0">
                <span className="text-xs text-muted-foreground mr-2">
                    {new Date(log.timestamp).toLocaleTimeString()}
                </span>
                <span className={log.type === 'error' ? 'text-red-400' : 'text-foreground'}>
                    {log.message}
                </span>
                </div>
            ))}
            <div ref={bottomRef} />
            </div>
        </ScrollArea>
      </CardContent>
    </Card>
  )
}
