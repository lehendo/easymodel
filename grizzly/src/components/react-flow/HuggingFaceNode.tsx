"use client";

import { useState, useEffect } from "react";
import { Handle, Position, useReactFlow } from "@xyflow/react";
import * as hub from "@huggingface/hub"; // Import Hugging Face Hub library

import { useToast } from "@/hooks/use-toast";

export default function HuggingFaceNode({ data, id }) {
  const [modelId, setModelId] = useState(data.modelId || "");
  const { toast } = useToast({ variant: "destructive" });
  const { setNodes } = useReactFlow();

  // Update the node's data when modelId changes - use setNodes to trigger React Flow updates
  useEffect(() => {
    // Only update if the value actually changed
    if (data.modelId !== modelId) {
      setNodes((nds) =>
        nds.map((node) => {
          if (node.id === id) {
            return {
              ...node,
              data: {
                ...node.data,
                modelId: modelId,
              },
            };
          }
          return node;
        })
      );
    }
  }, [modelId, id, setNodes, data.modelId]);

  // Function to extract the model ID from a full URL or accept it directly
  const extractModelId = (input) => {
    try {
      const url = new URL(input);
      if (url.hostname === "huggingface.co") {
        return url.pathname.slice(1); // Remove the leading "/"
      }
    } catch {
      // If it's not a valid URL, assume it's a raw modelId
    }
    return input; // Return the input as-is if it's not a URL
  };

  // Function to validate model existence using Hugging Face Hub library
  const validateModelId = async () => {
    const trimmedModelId = extractModelId(modelId);
    try {
      await hub.modelInfo({ name: trimmedModelId });
      setModelId(trimmedModelId); // Update the display to show only the model ID
    } catch {
      toast({
        variant: "destructive",
        title: "Uh oh! Something went wrong.",
        description: "Invalid HuggingFace model URL given.",
      });
      setModelId(""); // Clear the input on error
    }
  };

  // Handle key press in the input field
  const handleKeyPress = (event) => {
    if (event.key === "Enter") {
      validateModelId();
    }
  };

  const handleChange = (event) => {
    setModelId(event.target.value);
  };

  return (
    <div
      className="rounded-lg border-2 border-gray-300 p-4 shadow-md"
      style={{ backgroundColor: "#FFC107", color: "#333" }}
    >
      <Handle
        type="target"
        position={Position.Top}
        style={{ background: "#FF9900" }}
      />
      {/* <Handle
          type="target"
          position={Position.Left}
          style={{ background: "#FF9900" }}
          />
          <Handle
          type="target"
          position={Position.Right}
          style={{ background: "#FF9900" }}
          /> */}
      <div className="mb-2 text-center text-lg font-bold">
        HuggingFace Model
      </div>
      <input
        type="text"
        value={modelId} // Always display the trimmed model ID
        onChange={handleChange}
        onKeyDown={handleKeyPress}
        placeholder="Enter model URL or userName/modelName"
        className="nodrag w-full rounded border p-2 text-sm"
        style={{
          backgroundColor: "#fff", // White for clarity
          color: "#333", // Dark text
          borderColor: "#FF9900", // Hugging Face accent
        }}
      />
      <Handle
        type="source"
        position={Position.Bottom}
        style={{ background: "#FF9900" }}
      />
    </div>
  );
}
