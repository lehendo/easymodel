"use client";

import { useState, useRef, useEffect } from "react";
import { Button } from "../ui/button";
import { Card, CardContent } from "../ui/card";
import { ChevronRight, ChevronLeft } from "lucide-react";

const nodeTypes = [
  { type: "huggingFace", label: "HuggingFace Model" },
  { type: "dataset", label: "HuggingFace Dataset" },
  { type: "finetuning", label: "Finetuning Schema" },
];

const GRID_SIZE = 20;

export default function NodePalette({ onAddNode, getReactFlowBounds }) {
  const [isExpanded, setIsExpanded] = useState(true);
  const [placedNodes, setPlacedNodes] = useState<{ x: number; y: number }[]>(
    [],
  );

  useEffect(() => {
    setPlacedNodes([]);
  }, [getReactFlowBounds]);

  const onDragStart = (event: React.DragEvent, nodeType: string) => {
    event.dataTransfer.setData("application/reactflow", nodeType);
    event.dataTransfer.effectAllowed = "move";
  };

  const createNode = (nodeType: string) => {
    const bounds = getReactFlowBounds();
    if (!bounds) return;

    const width = bounds.width / 2;
    const height = bounds.height / 2;
    const gridWidth = Math.floor(width / GRID_SIZE);
    const gridHeight = Math.floor(height / GRID_SIZE);

    let position;
    let attempts = 0;
    const maxAttempts = gridWidth * gridHeight;

    do {
      const gridX = Math.floor(Math.random() * gridWidth) + 30;
      const gridY = Math.floor(Math.random() * gridHeight) + 20;
      position = {
        x: gridX * GRID_SIZE,
        y: gridY * GRID_SIZE,
      };
      attempts++;
    } while (
      placedNodes.some(
        (node) =>
          Math.abs(node.x - position.x) < GRID_SIZE &&
          Math.abs(node.y - position.y) < GRID_SIZE,
      ) &&
      attempts < maxAttempts
    );

    if (attempts === maxAttempts) {
      console.warn("Could not find a free position for the new node");
      return;
    }

    setPlacedNodes([...placedNodes, position]);

    const newNode = {
      id: `${nodeType}-${Date.now()}`,
      type: nodeType,
      position: position,
      data: { label: `${nodeType} Node` },
    };
    onAddNode(newNode);
  };


  return (
    <Card
      className={`absolute left-4 top-4 z-50 transition-all duration-300 ${
        isExpanded ? "w-64" : "w-12"
      }`}
    >
      <CardContent className="p-0">
        <div className="flex items-center">
          <Button
            variant="ghost"
            size="icon"
            className="h-12 w-12"
            onClick={() => setIsExpanded(!isExpanded)}
          >
            {isExpanded ? (
              <ChevronLeft className="h-4 w-4" />
            ) : (
              <ChevronRight className="h-4 w-4" />
            )}
          </Button>
        </div>
        {isExpanded && (
          <div className="p-2">
            <div
              className="overflow-y-auto"
              style={{
                display: "grid",
                gridTemplateRows: "repeat(4, auto)",
                maxHeight: "calc(4 * 44px)",
                rowGap: "0.5rem",
              }}
            >
              {nodeTypes.map((node) => {
                const customStyles =
                  node.type === "huggingFace"
                    ? { backgroundColor: "#FFC107", color: "#333" }
                    : node.type === "dataset"
                      ? { backgroundColor: "#FF9800", color: "#333" }
                      : {};

                return (
                  <Button
                    key={node.label}
                    variant="outline"
                    className="cursor-move"
                    draggable
                    onDragStart={(event) => onDragStart(event, node.type)}
                    onClick={() => createNode(node.type)}
                    style={customStyles}
                  >
                    {node.label}
                  </Button>
                );
              })}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
