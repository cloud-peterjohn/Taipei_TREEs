"use client"

import { Card } from "@/components/ui/card"
import type { ImageData } from "./green-coverage-analyzer"
import { XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Area, AreaChart } from "recharts"

interface CoverageChartProps {
  images: ImageData[]
}

export default function CoverageChart({ images }: CoverageChartProps) {
  // Sort images by year and prepare data for chart
  const chartData = [...images]
    .filter((img) => img.coverage !== undefined)
    .sort((a, b) => a.year - b.year)
    .map((img) => ({
      year: img.year,
      coverage: img.coverage,
    }))

  // Calculate percentage changes between years
  const dataWithChanges = chartData.map((item, index) => {
    if (index === 0) return { ...item, change: 0 }

    const prevCoverage = chartData[index - 1].coverage || 0
    const currentCoverage = item.coverage || 0
    const change = prevCoverage ? ((currentCoverage - prevCoverage) / prevCoverage) * 100 : 0

    return {
      ...item,
      change: Number.parseFloat(change.toFixed(1)),
    }
  })

  if (chartData.length < 2) {
    return (
      <div className="flex items-center justify-center h-64 border rounded-lg bg-gray-50">
        <p className="text-gray-500">At least two analyzed images are needed for visualization</p>
      </div>
    )
  }

  // Calculate average coverage
  const averageCoverage = chartData.reduce((sum, item) => sum + (item.coverage || 0), 0) / chartData.length

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload
      return (
        <div className="bg-white p-3 border rounded-lg shadow-lg">
          <div className="font-medium">{label}</div>
          <div className="text-green-600">Coverage: {payload[0].value}%</div>
          {data.change !== 0 && (
            <div className={data.change > 0 ? "text-green-600" : "text-red-500"}>
              Change: {data.change > 0 ? "+" : ""}
              {data.change}%
            </div>
          )}
        </div>
      )
    }
    return null
  }

  return (
    <div className="space-y-6">
      <div className="h-80 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={dataWithChanges} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id="colorCoverage" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#10B981" stopOpacity={0.8} />
                <stop offset="95%" stopColor="#10B981" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis dataKey="year" tick={{ fontSize: 12 }} tickLine={false} axisLine={false} />
            <YAxis
              domain={[0, 100]}
              tick={{ fontSize: 12 }}
              tickLine={false}
              axisLine={false}
              tickFormatter={(value) => `${value}%`}
            />
            <Tooltip content={<CustomTooltip />} />
            <ReferenceLine
              y={averageCoverage}
              stroke="#6B7280"
              strokeDasharray="3 3"
              label={{
                value: `Avg: ${averageCoverage.toFixed(1)}%`,
                position: "insideBottomRight",
                fill: "#6B7280",
                fontSize: 12,
              }}
            />
            <Area
              type="monotone"
              dataKey="coverage"
              stroke="#10B981"
              strokeWidth={2}
              fillOpacity={1}
              fill="url(#colorCoverage)"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {dataWithChanges.slice(1).map((item, index) => (
          <Card
            key={index}
            className={`p-4 ${item.change > 0 ? "bg-green-50" : item.change < 0 ? "bg-red-50" : "bg-gray-50"}`}
          >
            <div className="text-sm text-gray-500">
              {dataWithChanges[index].year} → {item.year}
            </div>
            <div
              className={`text-lg font-medium ${item.change > 0 ? "text-green-600" : item.change < 0 ? "text-red-500" : "text-gray-700"}`}
            >
              {item.change > 0 ? "+" : ""}
              {item.change}%
            </div>
          </Card>
        ))}
      </div>
    </div>
  )
}

