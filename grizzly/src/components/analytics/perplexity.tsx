"use client"

import React, { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Toggle } from "@/components/ui/toggle"
import { Badge } from "@/components/ui/badge"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"

// Sample data
const data = [
  { epoch: 0, training: 100, validation: 120, baseline: 150 },
  { epoch: 5, training: 80, validation: 90, baseline: 150 },
  { epoch: 10, training: 60, validation: 70, baseline: 150 },
  { epoch: 15, training: 50, validation: 65, baseline: 150 },
  { epoch: 20, training: 40, validation: 55, baseline: 150 },
  { epoch: 25, training: 35, validation: 50, baseline: 150 },
  { epoch: 30, training: 30, validation: 45, baseline: 150 },
]

const annotations = [
  { epoch: 10, text: "Learning rate reduced by 50%" },
  { epoch: 15, text: "Validation perplexity diverged from training" },
]

export default function UnifiedPerplexityVisualization() {
  const [showBaseline, setShowBaseline] = useState(true)
  const [showTraining, setShowTraining] = useState(true)
  const [showValidation, setShowValidation] = useState(true)

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-background p-4 rounded-lg shadow-lg border border-border">
          <p className="font-bold">Epoch {label}</p>
          {payload.map((entry: any, index: number) => (
            <p key={index} style={{ color: entry.color }}>
              {entry.name}: {entry.value.toFixed(2)}
            </p>
          ))}
          {annotations.find(a => a.epoch === label) && (
            <p className="text-sm text-muted-foreground mt-2">
              Note: {annotations.find(a => a.epoch === label)?.text}
            </p>
          )}
        </div>
      )
    }
    return null
  }

  return (
    <Card className="w-full h-full">
      <CardHeader>
        <CardTitle>Unified Perplexity Visualization</CardTitle>
        <CardDescription>A multi-layered line graph showing perplexity over training epochs</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="mb-8 flex space-x-2">
          <Toggle pressed={showTraining} onPressedChange={setShowTraining}>
            <Badge variant={showTraining ? "default" : "outline"}>Training</Badge>
          </Toggle>
          <Toggle pressed={showValidation} onPressedChange={setShowValidation}>
            <Badge variant={showValidation ? "default" : "outline"}>Validation</Badge>
          </Toggle>
          <Toggle pressed={showBaseline} onPressedChange={setShowBaseline}>
            <Badge variant={showBaseline ? "default" : "outline"}>Baseline</Badge>
          </Toggle>
        </div>
        <ChartContainer
          config={{
            training: {
              label: "Training",
              color: "hsl(var(--chart-1))",
            },
            validation: {
              label: "Validation",
              color: "hsl(var(--chart-2))",
            },
            baseline: {
              label: "Baseline",
              color: "hsl(var(--chart-3))",
            },
          }}
          className="h-[400px] w-full"
        >
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="epoch" />
              <YAxis />
              <ChartTooltip content={<CustomTooltip />} />
              <Legend />
              {showTraining && (
                <Line type="monotone" dataKey="training" stroke="var(--color-training)" name="Training" />
              )}
              {showValidation && (
                <Line type="monotone" dataKey="validation" stroke="var(--color-validation)" name="Validation" />
              )}
              {showBaseline && (
                <Line type="monotone" dataKey="baseline" stroke="var(--color-baseline)" strokeDasharray="5 5" name="Baseline" />
              )}
            </LineChart>
          </ResponsiveContainer>
        </ChartContainer>
        <div className="mt-8 grid grid-cols-3 gap-4 text-center">
          <div>
            <p className="text-sm font-medium">Final Training Perplexity</p>
            <p className="text-2xl font-bold">{data[data.length - 1].training.toFixed(2)}</p>
            <p className="text-sm text-muted-foreground">
              {((data[0].training - data[data.length - 1].training) / data[0].training * 100).toFixed(2)}% improvement
            </p>
          </div>
          <div>
            <p className="text-sm font-medium">Final Validation Perplexity</p>
            <p className="text-2xl font-bold">{data[data.length - 1].validation.toFixed(2)}</p>
            <p className="text-sm text-muted-foreground">
              {((data[0].validation - data[data.length - 1].validation) / data[0].validation * 100).toFixed(2)}% improvement
            </p>
          </div>
          <div>
            <p className="text-sm font-medium">Baseline Perplexity</p>
            <p className="text-2xl font-bold">{data[data.length - 1].baseline.toFixed(2)}</p>
            <p className="text-sm text-muted-foreground">Constant baseline</p>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

