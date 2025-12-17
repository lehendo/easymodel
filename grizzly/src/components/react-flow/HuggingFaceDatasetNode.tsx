"use client";

import { useState, useEffect } from "react";
import { Handle, Position, useReactFlow } from "@xyflow/react";
import * as hub from "@huggingface/hub"; // Import Hugging Face Hub library

import { useToast } from "../../hooks/use-toast";

export default function HuggingFaceDatasetNode({ data, id }) {
  const [datasetId, setDatasetId] = useState(data.datasetId || "");
  const { toast } = useToast({ variant: "destructive" });
  const { setNodes } = useReactFlow();

  // Update the node's data when datasetId changes - use setNodes to trigger React Flow updates
  useEffect(() => {
    // Only update if the value actually changed
    if (data.datasetId !== datasetId) {
      setNodes((nds) =>
        nds.map((node) => {
          if (node.id === id) {
            return {
              ...node,
              data: {
                ...node.data,
                datasetId: datasetId,
              },
            };
          }
          return node;
        })
      );
    }
  }, [datasetId, id, setNodes, data.datasetId]);

  // Function to extract the dataset ID from a full URL or accept it directly
  const extractDatasetId = (input) => {
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
        // Remove "datasets/" prefix if present
        if (path.startsWith("datasets/")) {
          path = path.slice(9); // "datasets/" is 9 characters
        }
        return path;
      }
    } catch {
      // If it's not a valid URL, assume it's a raw datasetId
    }
    return trimmed; // Return the trimmed input as-is if it's not a URL
  };

  // Function to validate dataset existence using Hugging Face Hub library
  // NOTE: Validation is optional - backend will validate during training
  // This is just a convenience check and can be skipped if rate limited
  const validateDatasetId = async () => {
    if (!datasetId || !datasetId.trim()) {
      return; // Don't validate empty input
    }
    
    const trimmedDatasetId = extractDatasetId(datasetId);
    
    // If extraction resulted in empty, use original
    if (!trimmedDatasetId) {
      return;
    }
    
    // Update the display to show the extracted ID (even without validation)
    setDatasetId(trimmedDatasetId);
    
    // Try to validate, but don't block if it fails (rate limiting, network issues, etc.)
    try {
      const datasetInfo = await hub.datasetInfo({ name: trimmedDatasetId });
      console.log("Dataset validated:", trimmedDatasetId);
    } catch (error) {
      // Silently fail - validation is optional
      // Backend will handle actual validation during training
      console.log("Dataset validation skipped (rate limit or network issue):", error.message);
    }
  };

  // Handle key press in the input field - just extract ID, don't validate
  const handleKeyPress = (event) => {
    if (event.key === "Enter") {
      // Just extract and normalize the ID, don't validate
      const trimmedDatasetId = extractDatasetId(datasetId);
      if (trimmedDatasetId) {
        setDatasetId(trimmedDatasetId);
      }
    }
  };

  const handleChange = (event) => {
    setDatasetId(event.target.value);
  };

  return (
    <div
      className="rounded-lg border-2 border-gray-300 p-4 shadow-md bg-[#FF9800]"
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
        HuggingFace Dataset
      </div>
      <input
        type="text"
        value={datasetId} // Always display the trimmed dataset ID
        onChange={handleChange}
        onKeyDown={handleKeyPress}
        placeholder="Enter dataset URL or userName/datasetName"
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
