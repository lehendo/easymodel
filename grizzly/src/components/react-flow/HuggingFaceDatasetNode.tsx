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
    try {
      const url = new URL(input);
      if (url.hostname === "huggingface.co") {
        return url.pathname.slice(10); // Remove the leading "/"
      }
    } catch {
      // If it's not a valid URL, assume it's a raw datasetId
    }
    return input; // Return the input as-is if it's not a URL
  };

  // Function to validate dataset existence using Hugging Face Hub library
  const validateDatasetId = async () => {
    const trimmedDatasetId = extractDatasetId(datasetId);
    try {
      const datasetInfo = await hub.datasetInfo({ name: trimmedDatasetId });
      console.log("Dataset exists:", datasetInfo);
      setDatasetId(trimmedDatasetId); // Update the display to show only the dataset ID
    } catch {
      toast({
        variant: "destructive",
        title: "Uh oh! Something went wrong.",
        description: "Invalid HuggingFace dataset URL given.",
      });
      setDatasetId(""); // Clear the input on error
    }
  };

  // Handle key press in the input field
  const handleKeyPress = (event) => {
    if (event.key === "Enter") {
      validateDatasetId();
    }
  };

  const handleChange = (event) => {
    setDatasetId(event.target.value);
  };

  return (
    <div
      className="rounded-lg border-2 border-gray-300 p-4 shadow-md"
      style={{ backgroundColor: "#FF9800", color: "#333" }}
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
