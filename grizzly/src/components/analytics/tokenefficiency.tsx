"use client"

import React, { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../ui/tabs"
import { Button } from "../ui/button"
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ScatterChart, Scatter, ZAxis, ReferenceLine, Line } from 'recharts'
import { ChartContainer } from "../ui/chart"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from "../ui/dialog"

// Sample data for token efficiency
const tokenEfficiencyData = [
  { id: 1, task: "Summarization", efficiency: 0.75, originalText: "Long text about AI...", generatedText: "AI summary..." },
  { id: 2, task: "Question Answering", efficiency: 0.85, originalText: "What is machine learning?", generatedText: "Machine learning is..." },
  { id: 3, task: "Translation", efficiency: 0.9, originalText: "Hello, how are you?", generatedText: "Hola, ¿cómo estás?" },
  { id: 4, task: "Paraphrasing", efficiency: 0.7, originalText: "The quick brown fox...", generatedText: "A fast auburn canine..." },
  { id: 5, task: "Code Generation", efficiency: 0.8, originalText: "Create a function to...", generatedText: "def my_function():..." },
]

const tokenEfficiencyDataft = [
  { id: 1, task: "Summarization", efficiency: 0.85, originalText: "Long text about AI...", generatedText: "AI summary..." },
  { id: 2, task: "Question Answering", efficiency: 0.95, originalText: "What is machine learning?", generatedText: "Machine learning is..." },
  { id: 3, task: "Translation", efficiency: 0.95, originalText: "Hello, how are you?", generatedText: "Hola, ¿cómo estás?" },
  { id: 4, task: "Paraphrasing", efficiency: 0.8, originalText: "The quick brown fox...", generatedText: "A fast auburn canine..." },
  { id: 5, task: "Code Generation", efficiency: 0.9, originalText: "Create a function to...", generatedText: "def my_function():..." },
]

// Sample data for compression ratio
const compressionRatioData = [
  { id: 1, originalLength: 100, generatedLength: 25, task: "Summarization" },
  { id: 2, originalLength: 50, generatedLength: 60, task: "Question Answering" },
  { id: 3, originalLength: 30, generatedLength: 35, task: "Translation" },
  { id: 4, originalLength: 80, generatedLength: 75, task: "Paraphrasing" },
  { id: 5, originalLength: 120, generatedLength: 40, task: "Code Generation" },
]

const CustomBarTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload
    return (
      <div className="bg-background p-4 rounded-lg shadow-lg border border-border">
        <p className="font-bold">{data.task}</p>
        <p>Efficiency: {data.efficiency.toFixed(2)}</p>
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

export default function TokenEfficiencyVisualization() {
  const [selectedModel, setSelectedModel] = useState("pretrained")
  const [threshold, setThreshold] = useState(0.8)

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
              <ChartContainer className="h-[400px] w-full">
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
    <ChartContainer className="h-[450px] w-full">
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
  <Scatter
    name="Compression Ratio"
    data={compressionRatioData}
    fill="hsl(var(--primary))"
  />
  {/* Adding the y=x line without points */}
  <Scatter
    name="y = x Line"
    data={[
      { originalLength: 0, generatedLength: 0 },
      { originalLength: 120, generatedLength: 120 },
    ]}
    line
    shape={() => null} // Custom shape that renders nothing
    stroke="red"
    fill="red"
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
        <div className="mt-8">
          {tokenEfficiencyData.map((item) => (
            <Dialog key={item.id}>
              <DialogTrigger asChild>
                <Button variant="outline" className="mr-2 mb-2">
                  {item.task} Example
                </Button>
              </DialogTrigger>
              <DialogContent className="sm:max-w-[425px]">
                <DialogHeader>
                  <DialogTitle>{item.task}</DialogTitle>
                  <DialogDescription>
                    Efficiency: {item.efficiency.toFixed(2)}
                  </DialogDescription>
                </DialogHeader>
                <div className="grid gap-4 py-4">
                  <div>
                    <h4 className="mb-2 font-medium">Original Text:</h4>
                    <p className="text-sm">{item.originalText}</p>
                  </div>
                  <div>
                    <h4 className="mb-2 font-medium">Generated Text:</h4>
                    <p className="text-sm">{item.generatedText}</p>
                  </div>
                </div>
              </DialogContent>
            </Dialog>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
