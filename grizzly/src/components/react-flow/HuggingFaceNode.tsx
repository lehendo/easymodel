"use client";

import { useState, useEffect } from "react";
import { Handle, Position, useReactFlow } from "@xyflow/react";
import * as hub from "@huggingface/hub"; // Import Hugging Face Hub library

import { useToast } from "../../hooks/use-toast";

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
    if (!input || !input.trim()) return input;
    
    const trimmed = input.trim();
    
    try {
      const url = new URL(trimmed);
      if (url.hostname === "huggingface.co" || url.hostname === "www.huggingface.co") {
        let path = url.pathname;
        // Remove leading slash
        if (path.startsWith("/")) {
          path = path.slice(1);
        }
        // Remove "models/" prefix if present (some URLs have this)
        if (path.startsWith("models/")) {
          path = path.slice(7); // "models/" is 7 characters
        }
        return path;
      }
    } catch {
      // If it's not a valid URL, assume it's a raw modelId
    }
    return trimmed; // Return the trimmed input as-is if it's not a URL
  };

  // Function to validate model existence using Hugging Face Hub library
  // NOTE: Validation is optional - backend will validate during training
  // This is just a convenience check and can be skipped if rate limited
  const validateModelId = async () => {
    if (!modelId || !modelId.trim()) {
      return; // Don't validate empty input
    }
    
    const trimmedModelId = extractModelId(modelId);
    
    // If extraction resulted in empty, use original
    if (!trimmedModelId) {
      return;
    }
    
    // Update the display to show the extracted ID (even without validation)
    setModelId(trimmedModelId);
    
    // Try to validate, but don't block if it fails (rate limiting, network issues, etc.)
    try {
      await hub.modelInfo({ name: trimmedModelId });
      console.log("Model validated:", trimmedModelId);
    } catch (error) {
      // Silently fail - validation is optional
      // Backend will handle actual validation during training
      console.log("Model validation skipped (rate limit or network issue):", error.message);
    }
  };

  // Handle key press in the input field - just extract ID, don't validate
  const handleKeyPress = (event) => {
    if (event.key === "Enter") {
      // Just extract and normalize the ID, don't validate
      const trimmedModelId = extractModelId(modelId);
      if (trimmedModelId) {
        setModelId(trimmedModelId);
      }
    }
  };

  const handleChange = (event) => {
    setModelId(event.target.value);
  };

  return (
    <div
      className="rounded-lg border-2 border-gray-300 p-4 shadow-md bg-[#FFC107]"
      style={{ color: "#333" }}
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
        className="nodrag w-full rounded border p-2 text-sm border-[#FF9900]"
        style={{
          backgroundColor: "#fff",
          color: "#333",
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
