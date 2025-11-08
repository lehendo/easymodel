"use client"

import React, { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../ui/card"
import { Toggle } from "../ui/toggle"
import { Badge } from "../ui/badge"
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer, Tooltip } from 'recharts'
import { ChartContainer } from "../ui/chart"
import { Popover, PopoverContent, PopoverTrigger } from "../ui/popover"
import { Button } from "../ui/button"

// Sample data
const data = [
  { metric: 'Fluency', A: 80, B: 70 },
  { metric: 'Coherence', A: 85, B: 75 },
  { metric: 'Grammar', A: 65, B: 80 },
  { metric: 'Relevance', A: 90, B: 85 },
]

const qualitativeExamples = [
  {
    metric: 'Fluency',
    prompt: 'Explain quantum computing.',
    output: 'Quantum computing leverages quantum mechanics principles to process information. It uses qubits, which can exist in multiple states simultaneously, enabling complex calculations.',
    reference: 'Quantum computing is a type of computation that harnesses the collective properties of quantum states, such as superposition, interference, and entanglement, to perform calculations.',
    score: 80,
  },
  {
    metric: 'Coherence',
    prompt: 'Describe the water cycle.',
    output: 'The water cycle involves evaporation from bodies of water, condensation in clouds, precipitation as rain or snow, and collection in rivers, lakes, and oceans. This continuous process is driven by solar energy.',
    reference: 'The water cycle, also known as the hydrologic cycle, describes the continuous movement of water within the Earth and atmosphere. It involves processes such as evaporation, transpiration, condensation, precipitation, and runoff.',
    score: 85,
  },
  {
    metric: 'Grammar',
    prompt: 'Write a sentence about climate change.',
    output: 'Climate change is affecting global temperatures, causing extreme weather events, and impacting ecosystems worldwide.',
    reference: 'Climate change is a long-term alteration in global or regional climate patterns, often attributed to increased levels of atmospheric carbon dioxide produced by the use of fossil fuels.',
    score: 65,
  },
  {
    metric: 'Relevance',
    prompt: 'Explain the importance of renewable energy.',
    output: 'Renewable energy is crucial for reducing greenhouse gas emissions, mitigating climate change, and ensuring long-term energy sustainability. Sources like solar, wind, and hydroelectric power offer clean alternatives to fossil fuels.',
    reference: 'Renewable energy is important because it provides clean, sustainable power that reduces reliance on fossil fuels, lowers carbon emissions, and helps combat climate change while promoting energy independence.',
    score: 90,
  },
]

const CustomTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload
    return (
      <div className="bg-background p-4 rounded-lg shadow-lg border border-border">
        <p className="font-bold">{data.metric}</p>
        <p>Model Score: {data.A}</p>
        <p>Baseline Score: {data.B}</p>
      </div>
    )
  }
  return null
}

export default function UnifiedGQSVisualization() {
  const [showBaseline, setShowBaseline] = useState(true)

  const overallGQS = data.reduce((sum, item) => sum + item.A, 0) / data.length
  const baselineGQS = data.reduce((sum, item) => sum + item.B, 0) / data.length
  const improvement = ((overallGQS - baselineGQS) / baselineGQS * 100).toFixed(2)

  return (
    <Card className="w-full h-full">
      <CardHeader>
        <CardTitle>Unified GQS Visualization</CardTitle>
        <CardDescription>A multi-axis radar chart showing Generative Quality Scores</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="mb-4 flex space-x-2">
          <Toggle pressed={showBaseline} onPressedChange={setShowBaseline}>
            <Badge variant={showBaseline ? "default" : "outline"}>Show Baseline</Badge>
          </Toggle>
        </div>
        <div className="flex flex-col md:flex-row">
          <ChartContainer config={{}} className="h-[400px] w-full md:w-2/3">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart cx="50%" cy="50%" outerRadius="80%" data={data} margin={{ top: 5, right: 45, left: 45, bottom: 5 }}>
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
          <div className="w-full md:w-1/3 mt-4 md:mt-0 md:ml-4">
            <h3 className="text-lg font-semibold mb-2"></h3>
            <p className="text-2xl font-bold">Overall GQS: {overallGQS.toFixed(2)}</p>
            <p className="text-sm text-muted-foreground mb-4">
              {improvement}% improvement from baseline
            </p>
            {data.map((item) => (
              <div key={item.metric} className="mb-2">
                <div className="flex justify-between items-center">
                  <span>{item.metric}</span>
                  <span className="font-semibold">{item.A}</span>
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
        <div className="mt-2">
          <h3 className="text-lg font-semibold mb-2">Qualitative Examples</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {qualitativeExamples.map((example, index) => (
              <Popover key={index}>
                <PopoverTrigger asChild>
                  <Button variant="outline" className="w-full justify-start">
                    {example.metric}: {example.score}
                  </Button>
                </PopoverTrigger>
                <PopoverContent className="w-80">
                  <div className="space-y-2">
                    <h4 className="font-semibold">Prompt:</h4>
                    <p className="text-sm">{example.prompt}</p>
                    <h4 className="font-semibold">Output:</h4>
                    <p className="text-sm">{example.output}</p>
                    <h4 className="font-semibold">Reference:</h4>
                    <p className="text-sm">{example.reference}</p>
                  </div>
                </PopoverContent>
              </Popover>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}