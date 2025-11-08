"use client"

import React, { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Slider } from "@/components/ui/slider"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts'
import { ChartContainer } from "@/components/ui/chart"
import { ZoomIn, ZoomOut } from 'lucide-react'

// Simulated data for semantic consistency over text segments
const generateData = (segments: number) => {
  return Array.from({ length: segments }, (_, i) => ({
    segment: i + 1,
    // Constrain similarity value between 0 and 1
    similarity: Math.min(1, Math.max(0, Math.random() * 0.3 + 0.7 - Math.sin(i / 10) * 0.1)), 
  }))
}

const initialData = generateData(50)

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-background p-4 rounded-lg shadow-lg border border-border">
        <p className="font-bold">Epoch: {label}</p>
        <p>Similarity: {payload[0].value.toFixed(3)}</p>
        <p className="mt-2 text-sm">
          {payload[0].value < 0.8 
            ? "Potential drift detected. The model may be losing context."
            : "Good semantic consistency in this epoch."}
        </p>
      </div>
    )
  }
  return null
}

export default function SemanticDriftAnalysis() {
  const [data, setData] = useState(initialData)
  const [threshold, setThreshold] = useState(0.8)
  const [zoomLevel, setZoomLevel] = useState(1)

  // const handleZoomIn = () => {
  //   setZoomLevel(prev => Math.min(prev * 1.5, 4))
  // }

  // const handleZoomOut = () => {
  //   setZoomLevel(prev => Math.max(prev / 1.5, 1))
  // }

  const handleThresholdChange = (newThreshold: number[]) => {
    setThreshold(newThreshold[0])
  }

  return (
    <Card className="w-full h-full">
  <CardHeader>
    <CardTitle>Semantic Consistency & Drift Analysis</CardTitle>
    <CardDescription>Visualizing semantic coherence over epochs</CardDescription>
  </CardHeader>
  <CardContent>
    <div className="mb-4 flex justify-between items-center">
      <div className="flex items-center">
        <span className="mr-2">Similarity Threshold:</span>
        {/* <Slider
          value={[threshold]}
          onValueChange={handleThresholdChange}
          max={1}
          step={0.01}
          className="w-[200px]"
        /> */}
        <span className="ml-2 font-semibold">{threshold.toFixed(2)}</span>
      </div>
    </div>
    <ChartContainer className="h-[600px] w-full"> {/* Increase height here */}
      <ResponsiveContainer width="100%" height="100%"> {/* Ensure it fills the container */}
        <LineChart
          data={data}
          margin={{ top: 20, right: 10, left: 10, bottom: 15 }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            dataKey="segment" 
            label={{ value: 'Epoch', position: 'insideBottomRight', offset: -10 }} 
          />
          <YAxis 
            domain={[0, 1]} 
            ticks={[0.2, 0.4, 0.6, 0.8, 1.0]} 
            label={{ value: 'Semantic Similarity', angle: -90, position: 'insideLeft' }} 
          />
          <Tooltip content={<CustomTooltip />} />
          <Line 
            type="monotone" 
            dataKey="similarity" 
            stroke="hsl(var(--primary))" 
            strokeWidth={2} 
            dot={false} 
          />
          <ReferenceLine y={threshold} stroke="hsl(var(--destructive))" strokeWidth={2} />
        </LineChart>
      </ResponsiveContainer>
    </ChartContainer>
    <div className="mt-8">
      <p className="text-sm text-muted-foreground">
        The red solid line represents the similarity threshold. 
        Epochs below this line may indicate semantic drift or loss of context.
      </p>
    </div>
  </CardContent>
</Card>
  )
}
