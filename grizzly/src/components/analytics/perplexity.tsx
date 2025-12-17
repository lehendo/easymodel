"use client"

import React, { useState, useMemo } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../ui/card"
import { Toggle } from "../ui/toggle"
import { Badge } from "../ui/badge"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "../ui/chart"

// Sample data (fallback)
const sampleData = [
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

interface PerplexityData {
  epochs?: number[];
  training?: number[];
  validation?: number[];
  baseline?: number;
}

interface Props {
  data?: PerplexityData;
}

export default function UnifiedPerplexityVisualization({ data: propData }: Props) {
  const [showBaseline, setShowBaseline] = useState(true)
  const [showTraining, setShowTraining] = useState(true)
  const [showValidation, setShowValidation] = useState(true)

  // Transform data for chart
  const data = useMemo(() => {
    if (!propData || !propData.epochs || propData.epochs.length === 0) {
      return sampleData;
    }

    // Use baseline from data, or compute from first validation if not set
    let baseline = propData.baseline;
    if (baseline === null || baseline === undefined) {
      // Use first validation value as baseline if not explicitly set
      baseline = propData.validation?.[0] || 150;
    }
    
    // Create data points for all epochs
    const chartData = propData.epochs.map((epoch, index) => ({
      epoch,
      training: propData.training?.[index] || 0,
      validation: propData.validation?.[index] || 0,
      baseline: baseline, // Constant baseline across all epochs
    }));
    
    // Debug log to verify data
    console.log("[Perplexity Chart] Data points:", chartData.length, "epochs:", propData.epochs, 
                "training:", propData.training, "validation:", propData.validation, "baseline:", baseline);
    
    return chartData;
  }, [propData])

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
          <Toggle pressed={showTraining} onPressedChange={setShowTraining} className="focus-visible:ring-0 data-[state=on]:bg-transparent">
            <Badge variant={showTraining ? "default" : "outline"}>Training</Badge>
          </Toggle>
          <Toggle pressed={showValidation} onPressedChange={setShowValidation} className="focus-visible:ring-0 data-[state=on]:bg-transparent">
            <Badge variant={showValidation ? "default" : "outline"}>Validation</Badge>
          </Toggle>
          <Toggle pressed={showBaseline} onPressedChange={setShowBaseline} className="focus-visible:ring-0 data-[state=on]:bg-transparent">
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
          key={`perplexity-${data.length}-${data[data.length - 1]?.epoch || 0}`}
        >
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 30 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="epoch" 
                label={{ value: 'Epoch', position: 'insideBottomRight', offset: -10 }} 
              />
              <YAxis 
                label={{ value: 'Perplexity', angle: -90, position: 'insideLeft' }} 
              />
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

