"use client"

import React, { useState, useMemo } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../ui/card"
import { Toggle } from "../ui/toggle"
import { Badge } from "../ui/badge"
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer, Tooltip } from 'recharts'
import { ChartContainer } from "../ui/chart"
import { Popover, PopoverContent, PopoverTrigger } from "../ui/popover"
import { Button } from "../ui/button"

// Sample data (fallback)
const sampleData = [
  { metric: 'Fluency', A: 80, B: 70 },
  { metric: 'Coherence', A: 85, B: 75 },
  { metric: 'Grammar', A: 65, B: 80 },
  { metric: 'Relevance', A: 90, B: 85 },
]

interface GQSData {
  metrics?: string[];
  model?: number[];
  baseline?: number[];
  examples?: any[];
}

interface Props {
  data?: GQSData;
}

const CustomTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload
    return (
      <div className="bg-background p-4 rounded-lg shadow-lg border border-border">
        <p className="font-bold">{data.metric}</p>
        <p>Model Score: {data.A.toFixed(2)}</p>
        <p>Baseline Score: {data.B.toFixed(2)}</p>
      </div>
    )
  }
  return null
}

export default function UnifiedGQSVisualization({ data: propData }: Props) {
  const [showBaseline, setShowBaseline] = useState(true)

  // Transform data for chart
  const data = useMemo(() => {
    if (!propData || !propData.metrics || propData.metrics.length === 0) {
      return sampleData;
    }

    // propData.model can be:
    // - Array of arrays (one per epoch): [[fluency, coherence, grammar, relevance], ...]
    // - Single array: [fluency, coherence, grammar, relevance]
    let latestModelGQS = [];
    if (Array.isArray(propData.model)) {
      if (propData.model.length > 0 && Array.isArray(propData.model[0])) {
        // Array of arrays - get latest epoch
        latestModelGQS = propData.model[propData.model.length - 1] || [];
      } else {
        // Single array
        latestModelGQS = propData.model;
      }
    }
    
    const baselineGQS = Array.isArray(propData.baseline) ? propData.baseline : [];

    console.log("[GQS Chart] Data structure:", {
      hasModel: !!propData.model,
      modelType: Array.isArray(propData.model) ? (Array.isArray(propData.model[0]) ? "array of arrays" : "single array") : "not array",
      modelLength: Array.isArray(propData.model) ? propData.model.length : 0,
      fullModelArray: propData.model,
      latestModelGQS,
      baselineGQS,
      metrics: propData.metrics,
      computedData: propData.metrics.map((metric, index) => ({
        metric,
        A: Array.isArray(latestModelGQS) && latestModelGQS.length > index ? (latestModelGQS[index] || 0) : 0,
        B: Array.isArray(baselineGQS) && baselineGQS.length > index ? (baselineGQS[index] || 0) : 0,
      }))
    });

    return propData.metrics.map((metric, index) => ({
      metric,
      A: Array.isArray(latestModelGQS) && latestModelGQS.length > index ? (latestModelGQS[index] || 0) : 0,
      B: Array.isArray(baselineGQS) && baselineGQS.length > index ? (baselineGQS[index] || 0) : 0,
    }));
  }, [propData])

  const overallGQS = data.reduce((sum, item) => sum + item.A, 0) / data.length
  const baselineGQS = data.reduce((sum, item) => sum + item.B, 0) / data.length
  // Calculate improvement percentage, handle division by zero
  const improvement = baselineGQS > 0 
    ? ((overallGQS - baselineGQS) / baselineGQS * 100).toFixed(2)
    : "0.00"

  return (
    <Card className="w-full h-full">
      <CardHeader>
        <CardTitle>Unified GQS Visualization</CardTitle>
        <CardDescription>A multi-axis radar chart showing Generative Quality Scores</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="mb-4 flex space-x-2">
          <Toggle pressed={showBaseline} onPressedChange={setShowBaseline} className="focus-visible:ring-0 data-[state=on]:bg-transparent">
            <Badge variant={showBaseline ? "default" : "outline"}>Show Baseline</Badge>
          </Toggle>
        </div>
        <div className="flex flex-col md:flex-row gap-4">
          <ChartContainer config={{}} className="h-[500px] w-full md:w-2/3" key={`gqs-${data.length}-${overallGQS.toFixed(2)}`}>
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart cx="50%" cy="50%" outerRadius="80%" data={data} margin={{ top: 20, right: 45, left: 45, bottom: 20 }}>
                <PolarGrid />
                <PolarAngleAxis dataKey="metric" />
                {/* <PolarRadiusAxis angle={30} domain={[0, 100]} /> */}
                <Radar name="Model" dataKey="A" stroke="hsl(var(--primary))" fill="hsl(var(--primary))" fillOpacity={0.6} />
                {showBaseline && (
                  <Radar name="Baseline" dataKey="B" stroke="hsl(var(--primary))" fill="hsl(var(--secondary))" fillOpacity={0.3} />
                )}
                <Tooltip content={<CustomTooltip />} />
              </RadarChart>
            </ResponsiveContainer>
          </ChartContainer>
          <div className="w-full md:w-1/3 flex flex-col justify-center">
            <div className="mb-4">
              <p className="text-2xl font-bold">Overall GQS: {overallGQS.toFixed(2)}</p>
              <p className="text-sm text-muted-foreground">
                {improvement}% improvement from baseline
              </p>
            </div>
            <div className="space-y-3">
              {data.map((item) => (
                <div key={item.metric}>
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-sm">{item.metric}</span>
                    <span className="font-semibold text-sm">{item.A.toFixed(2)}</span>
                  </div>
                  <div className="w-full bg-secondary h-2 rounded-full">
                    <div
                      className="bg-primary h-2 rounded-full"
                      style={{ width: `${item.A}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}