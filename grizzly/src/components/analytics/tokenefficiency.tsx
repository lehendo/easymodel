"use client"

import React, { useState, useMemo } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../ui/tabs"
import { Button } from "../ui/button"
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ScatterChart, Scatter, ZAxis, ReferenceLine, Line } from 'recharts'
import { ChartContainer } from "../ui/chart"

// Sample data for token efficiency (fallback)
const sampleTokenEfficiencyData = [
  { id: 1, task: "Summarization", efficiency: 0.75, originalText: "Long text about AI...", generatedText: "AI summary..." },
  { id: 2, task: "Question Answering", efficiency: 0.85, originalText: "What is machine learning?", generatedText: "Machine learning is..." },
  { id: 3, task: "Translation", efficiency: 0.9, originalText: "Hello, how are you?", generatedText: "Hola, ¿cómo estás?" },
  { id: 4, task: "Paraphrasing", efficiency: 0.7, originalText: "The quick brown fox...", generatedText: "A fast auburn canine..." },
  { id: 5, task: "Code Generation", efficiency: 0.8, originalText: "Create a function to...", generatedText: "def my_function():..." },
]

const sampleTokenEfficiencyDataft = [
  { id: 1, task: "Summarization", efficiency: 0.85, originalText: "Long text about AI...", generatedText: "AI summary..." },
  { id: 2, task: "Question Answering", efficiency: 0.95, originalText: "What is machine learning?", generatedText: "Machine learning is..." },
  { id: 3, task: "Translation", efficiency: 0.95, originalText: "Hello, how are you?", generatedText: "Hola, ¿cómo estás?" },
  { id: 4, task: "Paraphrasing", efficiency: 0.8, originalText: "The quick brown fox...", generatedText: "A fast auburn canine..." },
  { id: 5, task: "Code Generation", efficiency: 0.9, originalText: "Create a function to...", generatedText: "def my_function():..." },
]

// Sample data for compression ratio
const sampleCompressionRatioData = [
  { id: 1, originalLength: 100, generatedLength: 25, task: "Summarization" },
  { id: 2, originalLength: 50, generatedLength: 60, task: "Question Answering" },
  { id: 3, originalLength: 30, generatedLength: 35, task: "Translation" },
  { id: 4, originalLength: 80, generatedLength: 75, task: "Paraphrasing" },
  { id: 5, originalLength: 120, generatedLength: 40, task: "Code Generation" },
]

interface TokenEfficiencyData {
  tasks?: any[];
  pretrained?: any[];
  finetuned?: any[];
}

interface Props {
  data?: TokenEfficiencyData;
}

const CustomBarTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload
    return (
      <div className="bg-background p-4 rounded-lg shadow-lg border border-border">
        <p className="font-bold">{data.task}</p>
        <p>Efficiency: {(data.efficiency * 100).toFixed(2)}%</p>
        <p className="mt-2 text-sm">
          {data.efficiency >= 0.8 
            ? "High efficiency. Good token usage."
            : "Lower efficiency. Could be improved."}
        </p>
      </div>
    )
  }
  return null
}

const CustomScatterTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload
    return (
      <div className="bg-background p-4 rounded-lg shadow-lg border border-border">
        <p className="font-bold">{data.task}</p>
        <p>Original Length: {data.originalLength}</p>
        <p>Generated Length: {data.generatedLength}</p>
        <p>Compression Ratio: {(data.generatedLength / data.originalLength).toFixed(2)}</p>
      </div>
    )
  }
  return null
}

export default function TokenEfficiencyVisualization({ data: propData }: Props) {
  const [selectedModel, setSelectedModel] = useState("pretrained")
  const [threshold, setThreshold] = useState(0.8)

  // Transform data for charts
  const tokenEfficiencyData = useMemo(() => {
    if (propData?.pretrained && propData?.tasks && propData.pretrained.length > 0) {
      return propData.tasks.map((task: string, index: number) => ({
        id: index + 1,
        task: task,
        efficiency: propData.pretrained?.[index] || 0,
        originalText: `Sample text for ${task}...`,
        generatedText: `Generated output for ${task}...`,
      }));
    }
    return sampleTokenEfficiencyData;
  }, [propData])

  const tokenEfficiencyDataft = useMemo(() => {
    if (propData?.finetuned && propData?.tasks && propData.finetuned.length > 0) {
      const data = propData.tasks.map((task: string, index: number) => ({
        id: index + 1,
        task: task,
        efficiency: propData.finetuned?.[index] || 0,
        originalText: `Sample text for ${task}...`,
        generatedText: `Generated output for ${task}...`,
      }));
      console.log("[Token Efficiency] Fine-tuned data:", data, "from propData.finetuned:", propData.finetuned);
      return data;
    }
    console.log("[Token Efficiency] No fine-tuned data, using sample data");
    return sampleTokenEfficiencyDataft;
  }, [propData])

  const compressionRatioData = useMemo(() => {
    if (propData?.tasks && propData?.pretrained && propData?.finetuned) {
      const baseData = propData.tasks.map((task: string, index: number) => {
        // Compute compression ratio from actual efficiency values
        // Compression ratio = output_length / input_length
        // Lower ratio = better compression
        const pretrainedEff = (propData.pretrained && propData.pretrained[index]) || 0.7;
        const finetunedEff = (propData.finetuned && propData.finetuned[index]) || 0.75;
        
        // Use efficiency to estimate compression ratio
        // Efficiency represents useful tokens / total tokens
        // For compression: ratio = 1 / efficiency (inverse relationship)
        // Higher efficiency = lower compression ratio (better compression)
        const baseLength = 100;
        const originalLength = baseLength;
        
        // Compute generated length based on efficiency
        // More efficient = fewer tokens needed = better compression
        // Use inverse relationship: generatedLength = originalLength * (1 - efficiency * 0.8)
        // This ensures different tasks show different compression ratios
        const pretrainedRatio = 1 - (pretrainedEff * 0.8);
        const finetunedRatio = 1 - (finetunedEff * 0.8);
        
        // Use finetuned for display (shows current state)
        const generatedLength = Math.max(20, Math.round(originalLength * finetunedRatio));
        
        return {
          id: index + 1,
          originalLength: originalLength,
          generatedLength: generatedLength,
          task: task,
        };
      });

      // Add jittering to prevent overlapping points
      // Group points by their coordinates and add small offsets
      const jitteredData = baseData.map((point, index) => {
        // Find points with same or very similar coordinates
        const similarPoints = baseData.filter((p, i) => 
          i !== index &&
          Math.abs(p.originalLength - point.originalLength) < 2 &&
          Math.abs(p.generatedLength - point.generatedLength) < 2
        );

        // If there are similar points, add jitter
        if (similarPoints.length > 0) {
          // Create a circular jitter pattern for overlapping points
          const angle = (index * 2 * Math.PI) / baseData.length;
          const jitterRadius = 3; // Small offset radius
          const jitterX = Math.cos(angle) * jitterRadius;
          const jitterY = Math.sin(angle) * jitterRadius;
          
          return {
            ...point,
            originalLength: point.originalLength + jitterX,
            generatedLength: point.generatedLength + jitterY,
          };
        }
        
        return point;
      });

      return jitteredData;
    }
    return sampleCompressionRatioData;
  }, [propData])

  // Choose the data based on the selected model
  const currentData = selectedModel === "pretrained" ? tokenEfficiencyData : tokenEfficiencyDataft

  return (
    <Card className="w-full h-full">
      <CardHeader>
        <CardTitle>Token Efficiency & Compression Ratio</CardTitle>
        <CardDescription>Analyzing token usage efficiency and text compression across tasks</CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="efficiency" className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="efficiency">Token Efficiency</TabsTrigger>
            <TabsTrigger value="compression">Compression Ratio</TabsTrigger>
          </TabsList>
          <TabsContent value="efficiency">
            <div className="mt-4">
              <div className="mb-4 flex justify-between items-center">
                <div>
                  <Button 
                    variant={selectedModel === "pretrained" ? "default" : "outline"} 
                    onClick={() => setSelectedModel("pretrained")}
                    className="mr-2"
                  >
                    Pre-trained
                  </Button>
                  <Button 
                    variant={selectedModel === "finetuned" ? "default" : "outline"} 
                    onClick={() => setSelectedModel("finetuned")}
                  >
                    Fine-tuned
                  </Button>
                </div>
              </div>
              <ChartContainer 
                config={{
                  efficiency: {
                    label: "Token Efficiency",
                    color: "hsl(var(--destructive))",
                  },
                }}
                className="h-[550px] w-full" 
                key={`token-eff-${selectedModel}-${currentData.length}-${currentData[0]?.efficiency || 0}`}
              >
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={currentData}
                    margin={{ top: 10, right: 20, left: 20, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="task" />
                    <YAxis 
            domain={[0, 1]} 
            label={{ value: 'Token Efficiency', angle: -90, position: 'insideLeft' }} 
          />
                    <Tooltip content={<CustomBarTooltip />} />
                    <Bar dataKey="efficiency" fill="hsl(var(--destructive))" fillOpacity={0.3} />
                    <ReferenceLine y={threshold} stroke="hsl(var(--destructive))" strokeWidth={2} />
                  </BarChart>
                </ResponsiveContainer>
              </ChartContainer>
              <div className="mt-4">
                <p className="text-sm text-muted-foreground">
                  The red line represents the target efficiency threshold (0.8). 
                  Bars above this line indicate good token usage efficiency.
                </p>
              </div>
            </div>
          </TabsContent>
          <TabsContent value="compression">
  <div className="mt-4">
    <ChartContainer 
      config={{
        compression: {
          label: "Compression Ratio",
          color: "hsl(var(--chart-1))",
        },
      }}
      className="h-[600px] w-full" 
      key={`compression-${compressionRatioData.length}-${compressionRatioData[0]?.originalLength || 0}`}
    >
      <ResponsiveContainer width="100%" height="100%">
      <ScatterChart
  margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
>
  <CartesianGrid />
  <XAxis
    type="number"
    dataKey="originalLength"
    name="Original Length"
    unit=" tokens"
    ticks={[0, 30, 60, 90, 120]}
  />
  <YAxis
    type="number"
    dataKey="generatedLength"
    name="Generated Length"
    unit=" tokens"
    ticks={[0, 30, 60, 90, 120]}
  />
  <ZAxis type="category" dataKey="task" name="Task" />
  <Tooltip content={<CustomScatterTooltip />} />
  {compressionRatioData.map((point, index) => {
    // Use different colors for each point to make them distinguishable
    const colors = [
      "hsl(var(--chart-1))",
      "hsl(var(--chart-2))",
      "hsl(var(--chart-3))",
      "hsl(var(--chart-4))",
      "hsl(var(--chart-5))",
    ];
    const color = colors[index % colors.length];
    
    return (
      <Scatter
        key={`scatter-${point.id}-${index}`}
        name={point.task}
        data={[point]}
        fill={color}
        stroke={color}
        strokeWidth={2}
        fillOpacity={0.8}
      />
    );
  })}
  {/* Diagonal reference line y=x */}
  <ReferenceLine 
    segment={[{ x: 0, y: 0 }, { x: 120, y: 120 }]}
    stroke="red" 
    strokeWidth={2} 
    strokeDasharray="3 3"
  />
</ScatterChart>


      </ResponsiveContainer>
    </ChartContainer>
    <div className="mt-4">
      <p className="text-sm text-muted-foreground">
        Points below the diagonal represent effective compression, 
        where the generated text is shorter than the original.
      </p>
    </div>
  </div>
</TabsContent>


        </Tabs>
      </CardContent>
    </Card>
  )
}
