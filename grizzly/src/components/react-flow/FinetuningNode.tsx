"use client";

import { useState, useEffect, useMemo } from "react";
import { Handle, Position, useReactFlow, useStore } from "@xyflow/react";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { useToast } from "../../hooks/use-toast";
import { api } from "../../trpc/react";
import { Loader2, CheckCircle2, XCircle, AlertCircle } from "lucide-react";
import * as hub from "@huggingface/hub";
import { env } from "../../env";

export default function FinetuningNode({ data, id }: { data: any; id: string }) {
  const { getEdges, getNodes } = useReactFlow();
  const { toast } = useToast();
  const [outputSpace, setOutputSpace] = useState(data.outputSpace || "");
  const [numEpochs, setNumEpochs] = useState(data.numEpochs || 1);
  const [batchSize, setBatchSize] = useState(data.batchSize || 8);
  const [maxLength, setMaxLength] = useState(data.maxLength || 128);
  const [subsetSize, setSubsetSize] = useState(data.subsetSize || 1000);
  const [taskType, setTaskType] = useState(data.taskType || "generation");
  const [textField, setTextField] = useState(data.textField || "");
  const [labelField, setLabelField] = useState(data.labelField || "");
  const [huggingFaceToken, setHuggingFaceToken] = useState("");
  const [trainingStatus, setTrainingStatus] = useState<"idle" | "training" | "success" | "error">("idle");
  const [errorMessage, setErrorMessage] = useState("");
  const [isInferringColumns, setIsInferringColumns] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [trainingStartTime, setTrainingStartTime] = useState<number | null>(null);
  const [analyticsData, setAnalyticsData] = useState<any>(null);
  const [progressMessage, setProgressMessage] = useState("");
  const [currentEpoch, setCurrentEpoch] = useState<number | null>(null);
  const [totalEpochs, setTotalEpochs] = useState<number | null>(null);
  const [backendJobId, setBackendJobId] = useState<string | null>(null);
  const [eventSource, setEventSource] = useState<EventSource | null>(null);

  // Persist HF token only for current browser session (never stored in database)
  useEffect(() => {
    if (typeof window === "undefined") return;
    const storedToken = window.sessionStorage.getItem("easymodel:hfToken");
    if (storedToken) {
      setHuggingFaceToken(storedToken);
    }
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") return;
    if (huggingFaceToken) {
      window.sessionStorage.setItem("easymodel:hfToken", huggingFaceToken);
    } else {
      window.sessionStorage.removeItem("easymodel:hfToken");
    }
  }, [huggingFaceToken]);

  const sanitizedApiBaseUrl = env.NEXT_PUBLIC_EASYMODEL_API_URL
    ? env.NEXT_PUBLIC_EASYMODEL_API_URL.replace(/\/$/, "")
    : undefined;
  const isHostedEnvironment =
    typeof window !== "undefined" &&
    !["localhost", "127.0.0.1"].includes(window.location.hostname);

  // Subscribe to edges and nodes changes to make connection detection reactive
  const edges = useStore((store) => store.edges);
  const nodes = useStore((store) => store.nodes);
  
  // Get connected nodes - now reactive to edge/node changes
  // Check edges connecting to either top or bottom target handles
  const connectedNodes = useMemo(() => {
    // Get all edges that target this node (from either top or bottom handle)
    const incomingEdges = edges.filter((e) => e.target === id);

    const modelNode = nodes.find(
      (n) =>
        incomingEdges.some((e) => e.source === n.id) &&
        n.type === "huggingFace"
    );
    const datasetNode = nodes.find(
      (n) =>
        incomingEdges.some((e) => e.source === n.id) &&
        n.type === "dataset"
    );

    return {
      model: modelNode,
      dataset: datasetNode,
    };
  }, [id, edges, nodes]);

  // Auto-infer columns when dataset is connected and has a datasetId
  useEffect(() => {
    const inferColumns = async () => {
      const datasetId = connectedNodes.dataset?.data?.datasetId;
      if (!datasetId || typeof datasetId !== 'string' || !datasetId.trim() || isInferringColumns) return;
      
      // Only infer if textField is empty or we just connected a new dataset
      if ((textField && textField.trim()) && (labelField && labelField.trim())) return;
      
      setIsInferringColumns(true);
      try {
        const datasetInfo = await hub.datasetInfo({ name: datasetId.trim() });
        
        // Get column names from dataset info - try different possible structures
        let columnNames: string[] = [];
        if (datasetInfo && typeof datasetInfo === 'object') {
          // Try to get features/columns from the dataset info
          const info = datasetInfo as any;
          if (info.features && typeof info.features === 'object') {
            columnNames = Object.keys(info.features);
          } else if (info.default && info.default.features) {
            columnNames = Object.keys(info.default.features);
          }
        }
        
        if (columnNames.length > 0) {
          // Auto-detect text field (look for common text column names)
          const textCandidates = columnNames.filter(col => 
            ['text', 'sentence', 'content', 'input', 'review', 'comment', 'article'].some(
              keyword => col.toLowerCase().includes(keyword)
            )
          );
          const detectedTextField = textCandidates[0] || columnNames.find(col => 
            col.toLowerCase() !== 'label' && col.toLowerCase() !== 'id'
          ) || columnNames[0];
          
          // Auto-detect label field (look for common label column names)
          const labelCandidates = columnNames.filter(col => 
            ['label', 'target', 'class', 'category', 'sentiment', 'rating'].some(
              keyword => col.toLowerCase().includes(keyword)
            )
          );
          const detectedLabelField = labelCandidates[0] || (taskType !== "generation" ? columnNames.find(col => 
            col.toLowerCase() !== detectedTextField?.toLowerCase() && col.toLowerCase() !== 'id'
          ) : null);
          
          // Update fields if they're empty
          if (detectedTextField && (!textField || !textField.trim())) {
            setTextField(detectedTextField);
          }
          if (taskType !== "generation" && detectedLabelField && (!labelField || !labelField.trim())) {
            setLabelField(detectedLabelField);
          }
          
          toast({
            title: "Columns Inferred",
            description: `Detected text field: ${detectedTextField}${detectedLabelField ? `, label field: ${detectedLabelField}` : ''}`,
          });
        }
      } catch (error) {
        console.warn("Failed to infer columns:", error);
        // Don't show error toast - just silently fail
      } finally {
        setIsInferringColumns(false);
      }
    };
    
    inferColumns();
  }, [connectedNodes.dataset?.data?.datasetId, taskType, textField, labelField, toast, isInferringColumns]);

  // Update data when form fields change
  useEffect(() => {
    data.outputSpace = outputSpace;
    data.numEpochs = numEpochs;
    data.batchSize = batchSize;
    data.maxLength = maxLength;
    data.subsetSize = subsetSize;
    data.taskType = taskType;
    data.textField = textField;
    data.labelField = labelField;
  }, [outputSpace, numEpochs, batchSize, maxLength, subsetSize, taskType, textField, labelField, data]);

  const projectId = typeof window !== "undefined" 
    ? window.location.pathname.split("/")[2] 
    : "";

  const executeFinetuningMutation = api.job.executeFinetuning.useMutation({
    onSuccess: async (data) => {
      // Start connecting to SSE for real-time progress
      if (data.backendJobId) {
        setBackendJobId(data.backendJobId);
        connectToProgressStream(data.backendJobId);
      } else {
        // Fallback to simulated progress if no backend job_id
        setTrainingStatus("training");
      }
    },
    onError: (error) => {
      setTrainingStatus("error");
      setTrainingProgress(0);
      setTrainingStartTime(null);
      setErrorMessage(error.message);
      toast({
        variant: "destructive",
        title: "Training Failed",
        description: error.message || "Failed to start fine-tuning job.",
      });
      // Reset status after 5 seconds
      setTimeout(() => {
        setTrainingStatus("idle");
        setErrorMessage("");
      }, 5000);
    },
  });

  // Connect to SSE progress stream
  const connectToProgressStream = (jobId: string) => {
    const apiUrl = sanitizedApiBaseUrl || (isHostedEnvironment ? undefined : "http://localhost:8000");
    if (!apiUrl) {
      setTrainingStatus("error");
      setErrorMessage("Backend URL is not configured. Please set NEXT_PUBLIC_EASYMODEL_API_URL to your backend.");
      toast({
        variant: "destructive",
        title: "Backend Not Configured",
        description: "Set NEXT_PUBLIC_EASYMODEL_API_URL to a reachable backend before starting training.",
      });
      return;
    }

    setTrainingStatus("training");
    setTrainingProgress(0);
    setProgressMessage("Connecting to training stream...");

    const es = new EventSource(`${apiUrl}/finetuning/progress/${jobId}`);
    setEventSource(es);

    es.onmessage = (event) => {
      try {
        const update = JSON.parse(event.data);
        
        // Update progress
        if (update.progress !== undefined) {
          setTrainingProgress(update.progress);
        }
        
        // Update message
        if (update.message) {
          setProgressMessage(update.message);
        }
        
        // Update epoch info
        if (update.epoch) {
          setCurrentEpoch(update.epoch);
        }
        if (update.total_epochs) {
          setTotalEpochs(update.total_epochs);
        }
        
        // Handle completion
        if (update.stage === "completed") {
          setTrainingStatus("success");
          es.close();
          setEventSource(null);
          
          toast({
            title: "Training Completed",
            description: update.message || "Fine-tuning completed successfully.",
          });
          
          // Fetch analytics after training completes
          fetchAnalytics();
          
          // Keep success state visible until user manually closes it
        }
        
        // Handle cancellation
        if (update.stage === "cancelled" || update.stage === "cancelling") {
          setTrainingStatus("idle");
          setProgressMessage("");
          setTrainingProgress(0);
          setTrainingStartTime(null);
          setCurrentEpoch(null);
          setTotalEpochs(null);
          setBackendJobId(null);
          es.close();
          setEventSource(null);
          
          toast({
            title: "Training Cancelled",
            description: update.message || "Training has been cancelled.",
          });
        }
        
        // Handle errors
        if (update.stage === "error") {
          setTrainingStatus("error");
          setErrorMessage(update.message || "Training failed");
          setTrainingProgress(0);
          setTrainingStartTime(null);
          setCurrentEpoch(null);
          setTotalEpochs(null);
          setProgressMessage("");
          setBackendJobId(null);
          es.close();
          setEventSource(null);
          
          toast({
            variant: "destructive",
            title: "Training Failed",
            description: update.message || "Training failed.",
          });
          
          // Auto-reset error state after 5 seconds
          setTimeout(() => {
            setTrainingStatus("idle");
            setErrorMessage("");
          }, 5000);
        }
      } catch (error) {
        console.error("Error parsing progress update:", error);
      }
    };

    es.onerror = (error) => {
      console.error("SSE connection error:", error);
      es.close();
      setEventSource(null);
      // Fallback to simulated progress on error
      if (trainingStatus === "training") {
        // Keep training status but show warning
        setProgressMessage("Connection lost, progress may be inaccurate");
      }
    };
  };

  // Cancel training mutation
  const cancelTrainingMutation = api.job.cancelTraining.useMutation({
    onSuccess: (data) => {
      // Close SSE connection
      if (eventSource) {
        eventSource.close();
        setEventSource(null);
      }
      
      if (data.status === "not_found") {
        // Job already completed or doesn't exist
        setTrainingStatus("idle");
        setProgressMessage("");
        toast({
          title: "Job Not Found",
          description: "The training job may have already completed or doesn't exist.",
        });
      } else {
        setProgressMessage("Cancelling training...");
        toast({
          title: "Cancelling Training",
          description: "Training cancellation requested.",
        });
      }
    },
    onError: (error) => {
      // If it's a 404, the job might have already completed - that's okay
      if (error.message.includes("404") || error.message.includes("not found")) {
        setTrainingStatus("idle");
        setProgressMessage("");
        toast({
          title: "Job Not Found",
          description: "The training job may have already completed.",
        });
      } else {
        toast({
          variant: "destructive",
          title: "Cancel Failed",
          description: error.message || "Failed to cancel training.",
        });
      }
    },
  });

  // Cancel training
  const handleCancel = () => {
    if (!backendJobId) return;
    cancelTrainingMutation.mutate({ backendJobId });
  };

  // Reset training state (close success/analytics card)
  const handleResetTraining = () => {
    setTrainingStatus("idle");
    setTrainingProgress(0);
    setTrainingStartTime(null);
    setCurrentEpoch(null);
    setTotalEpochs(null);
    setProgressMessage("");
    setBackendJobId(null);
    setAnalyticsData(null);
    setErrorMessage("");
  };

  // Fetch analytics after training
  const fetchAnalytics = async () => {
    try {
      const apiUrl = sanitizedApiBaseUrl || (isHostedEnvironment ? undefined : "http://localhost:8000");
      if (!apiUrl) return;
      const analyticsResponse = await fetch(`${apiUrl}/analytics`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model_name: outputSpace.trim() || connectedNodes.model?.data?.modelId || "",
          dataset_url: connectedNodes.dataset?.data?.datasetId || "",
          task_type: taskType,
        }),
      });
      
      if (analyticsResponse.ok) {
        const analytics = await analyticsResponse.json();
        setAnalyticsData(analytics.results);
      }
    } catch (error) {
      console.warn("Failed to fetch analytics:", error);
    }
  };

  // Check if nodes are connected (even if modelId/datasetId not filled yet)
  const isModelConnected = !!connectedNodes.model;
  const isDatasetConnected = !!connectedNodes.dataset;

  const handleTrain = () => {
    // Validation
    const modelId = connectedNodes.model?.data?.modelId;
    const datasetId = connectedNodes.dataset?.data?.datasetId;
    
    if (!isModelConnected || !modelId || typeof modelId !== 'string' || !modelId.trim()) {
      toast({
        variant: "destructive",
        title: "Validation Error",
        description: "Please connect a HuggingFace model node and enter a model name.",
      });
      return;
    }

    if (!isDatasetConnected || !datasetId || typeof datasetId !== 'string' || !datasetId.trim()) {
      toast({
        variant: "destructive",
        title: "Validation Error",
        description: "Please connect a HuggingFace dataset node and enter a dataset name.",
      });
      return;
    }

    if (!outputSpace.trim()) {
      toast({
        variant: "destructive",
        title: "Validation Error",
        description: "Please enter an output model name.",
      });
      return;
    }

    if (!textField.trim()) {
      toast({
        variant: "destructive",
        title: "Validation Error",
        description: "Please enter a text field name.",
      });
      return;
    }

    if (taskType !== "generation" && !labelField.trim()) {
      toast({
        variant: "destructive",
        title: "Validation Error",
        description: "Label field is required for non-generation tasks.",
      });
      return;
    }

    const numericFields = [
      { value: numEpochs, label: "Epochs" },
      { value: batchSize, label: "Batch size" },
      { value: maxLength, label: "Max length" },
      { value: subsetSize, label: "Subset size" },
    ];

    const invalidNumericField = numericFields.find(
      (field) => !Number.isFinite(field.value) || field.value <= 0,
    );

    if (invalidNumericField) {
      toast({
        variant: "destructive",
        title: "Validation Error",
        description: `${invalidNumericField.label} must be greater than 0.`,
      });
      return;
    }

    if (!huggingFaceToken.trim()) {
      toast({
        variant: "destructive",
        title: "Hugging Face Token Required",
        description: "Provide your Hugging Face API token to train with your own account.",
      });
      return;
    }

    if (
      isHostedEnvironment &&
      (!sanitizedApiBaseUrl || sanitizedApiBaseUrl.includes("localhost"))
    ) {
      toast({
        variant: "destructive",
        title: "Backend URL Missing",
        description:
          "Set NEXT_PUBLIC_EASYMODEL_API_URL to your deployed backend so training can reach it.",
      });
      return;
    }

    if (!projectId) {
      toast({
        variant: "destructive",
        title: "Validation Error",
        description: "Project ID is required.",
      });
      return;
    }

    setTrainingStatus("training");
    setErrorMessage("");
    setTrainingProgress(0);
    setTrainingStartTime(Date.now());
    setAnalyticsData(null);

    executeFinetuningMutation.mutate({
      projectId: projectId,
      modelName: modelId.trim(),
      datasets: [datasetId.trim()],
      outputSpace: outputSpace.trim(),
      numEpochs,
      batchSize,
      maxLength,
      subsetSize,
      taskType: taskType as "generation" | "classification" | "seq2seq" | "token_classification",
      textField: textField.trim(),
      labelField: labelField.trim() || undefined,
      apiKey: huggingFaceToken.trim(),
    });
  };
  
  // Get the model/dataset IDs, or show placeholder if connected but not filled
  const modelName = connectedNodes.model?.data?.modelId 
    ? connectedNodes.model.data.modelId 
    : isModelConnected 
    ? "Connected (enter model name)" 
    : "Not connected";
    
  const datasetName = connectedNodes.dataset?.data?.datasetId 
    ? connectedNodes.dataset.data.datasetId 
    : isDatasetConnected 
    ? "Connected (enter dataset name)" 
    : "Not connected";
  

  return (
    <div className="rounded-lg border-2 border-blue-300 bg-white p-4 pb-8 shadow-md min-w-[320px] relative">
      <Handle 
        type="target" 
        position={Position.Top}
        isConnectable={true}
        style={{ 
          background: '#3b82f6',
          width: '12px',
          height: '12px'
        }}
      />
      
      <div className="mb-3 text-center text-lg font-bold text-blue-700">
        Fine-tuning Node
      </div>

      {/* Connection Status */}
      <div className="mb-3 space-y-1 text-xs">
        <div className="flex items-center gap-2">
          <span className="font-semibold">Model:</span>
          <span className={isModelConnected ? "text-green-600" : "text-red-600"}>
            {isModelConnected ? (
              <>
                <CheckCircle2 className="mr-1 inline h-3 w-3" />
                {modelName}
              </>
            ) : (
              <>
                <XCircle className="mr-1 inline h-3 w-3" />
                {modelName}
              </>
            )}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="font-semibold">Dataset:</span>
          <span className={isDatasetConnected ? "text-green-600" : "text-red-600"}>
            {isDatasetConnected ? (
              <>
                <CheckCircle2 className="mr-1 inline h-3 w-3" />
                {datasetName}
              </>
            ) : (
              <>
                <XCircle className="mr-1 inline h-3 w-3" />
                {datasetName}
              </>
            )}
          </span>
        </div>
      </div>

      {/* Configuration Form */}
      <div className="space-y-2">
        <div>
          <label className="mb-1 block text-xs font-semibold">
            Hugging Face API Token
          </label>
          <Input
            type="password"
            value={huggingFaceToken}
            onChange={(e) => setHuggingFaceToken(e.target.value)}
            placeholder="hf_..."
            className="h-8 text-xs"
            disabled={trainingStatus === "training"}
          />
          <p className="mt-1 text-[10px] text-muted-foreground">
            Stored only in this browser session to run training under your Hugging Face account.
          </p>
        </div>
        <div>
          <label className="mb-1 block text-xs font-semibold">Output Model Name</label>
          <Input
            type="text"
            value={outputSpace}
            onChange={(e) => setOutputSpace(e.target.value)}
            placeholder="my-finetuned-model"
            className="h-8 text-xs"
            disabled={trainingStatus === "training"}
          />
        </div>

        <div className="grid grid-cols-2 gap-2">
          <div>
            <label className="mb-1 block text-xs font-semibold">Epochs</label>
            <Input
              type="number"
              value={numEpochs}
              onChange={(e) => setNumEpochs(parseInt(e.target.value) || 1)}
              min="1"
              className="h-8 text-xs"
              disabled={trainingStatus === "training"}
            />
          </div>
          <div>
            <label className="mb-1 block text-xs font-semibold">Batch Size</label>
            <Input
              type="number"
              value={batchSize}
              onChange={(e) => setBatchSize(parseInt(e.target.value) || 8)}
              min="1"
              className="h-8 text-xs"
              disabled={trainingStatus === "training"}
            />
          </div>
        </div>

        <div className="grid grid-cols-2 gap-2">
          <div>
            <label className="mb-1 block text-xs font-semibold">Max Length</label>
            <Input
              type="number"
              value={maxLength}
              onChange={(e) => setMaxLength(parseInt(e.target.value) || 128)}
              min="1"
              className="h-8 text-xs"
              disabled={trainingStatus === "training"}
            />
          </div>
          <div>
            <label className="mb-1 block text-xs font-semibold">Subset Size</label>
            <Input
              type="number"
              value={subsetSize}
              onChange={(e) => setSubsetSize(parseInt(e.target.value) || 1000)}
              min="1"
              className="h-8 text-xs"
              disabled={trainingStatus === "training"}
            />
          </div>
        </div>

        <div>
          <label className="mb-1 block text-xs font-semibold">Task Type</label>
          <select
            value={taskType}
            onChange={(e) => setTaskType(e.target.value)}
            className="flex h-8 w-full rounded-md border border-input bg-transparent px-3 py-1 text-xs shadow-sm transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
            disabled={trainingStatus === "training"}
          >
            <option value="generation">Generation</option>
            <option value="classification">Classification</option>
            <option value="seq2seq">Seq2Seq</option>
            <option value="token_classification">Token Classification</option>
          </select>
        </div>

        <div>
          <label className="mb-1 block text-xs font-semibold">Text Field</label>
          <Input
            type="text"
            value={textField}
            onChange={(e) => setTextField(e.target.value)}
            placeholder="text"
            className="h-8 text-xs"
            disabled={trainingStatus === "training"}
          />
        </div>

        {taskType !== "generation" && (
          <div>
            <label className="mb-1 block text-xs font-semibold">Label Field</label>
            <Input
              type="text"
              value={labelField}
              onChange={(e) => setLabelField(e.target.value)}
              placeholder="label"
              className="h-8 text-xs"
              disabled={trainingStatus === "training"}
            />
          </div>
        )}
      </div>

      {/* Progress Bar */}
      {/* {trainingStatus === "training" && (
        <div className="mt-2 space-y-1">
          <div className="flex justify-between text-xs text-gray-600">
            <span>
              {progressMessage || "Training in progress..."}
              {currentEpoch && totalEpochs && ` (Epoch ${currentEpoch}/${totalEpochs})`}
            </span>
            <span>{Math.round(trainingProgress)}%</span>
          </div>
          <div className="h-2 w-full overflow-hidden rounded-full bg-gray-200">
            <div
              className="h-full bg-blue-600 transition-all duration-300"
              style={{ width: `${trainingProgress}%` }}
            />
          </div>
        </div>
      )} */}

      {/* Error Message */}
      {errorMessage && (
        <div className="mt-2 flex items-center gap-2 rounded bg-red-50 p-2 text-xs text-red-600">
          <AlertCircle className="h-4 w-4" />
          <span>{errorMessage}</span>
        </div>
      )}

      {/* Success Message */}
      {trainingStatus === "success" && (
        <div className="mt-3 rounded-lg border-2 border-green-300 bg-green-50 p-3">
          <div className="mb-2 flex items-center justify-between">
            <div className="flex items-center gap-2 text-sm font-bold text-green-800">
              <CheckCircle2 className="h-4 w-4" />
              <span>Training Completed!</span>
            </div>
            <button
              onClick={handleResetTraining}
              className="text-green-600 hover:text-green-800"
              title="Close"
            >
              <XCircle className="h-4 w-4" />
            </button>
          </div>
          <div className="text-xs text-green-700 mb-2">
            Your model has been successfully trained and uploaded to Hugging Face!
          </div>
          
          {/* Analytics Card */}
          {analyticsData && (
            <div className="mt-2 space-y-1 text-xs text-green-700 border-t border-green-300 pt-2">
              <div className="font-semibold">Training Metrics:</div>
              {analyticsData.perplexity && (
                <div>Perplexity: {analyticsData.perplexity.toFixed(2)}</div>
              )}
              {analyticsData.accuracy && (
                <div>Accuracy: {(analyticsData.accuracy * 100).toFixed(2)}%</div>
              )}
              {analyticsData.f1_score && (
                <div>F1 Score: {analyticsData.f1_score.toFixed(2)}</div>
              )}
              {analyticsData.loss && (
                <div>Loss: {analyticsData.loss.toFixed(4)}</div>
              )}
              {!analyticsData.perplexity && !analyticsData.accuracy && !analyticsData.f1_score && (
                <div>Check your model on Hugging Face Hub for detailed metrics.</div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Train/Cancel Button */}
      <div className="mt-3 mb-2 flex gap-2">
        {trainingStatus === "training" ? (
          <Button
            onClick={handleCancel}
            variant="destructive"
            className="flex-1"
            size="sm"
          >
            <XCircle className="mr-2 h-4 w-4" />
            Cancel Training
          </Button>
        ) : trainingStatus === "success" ? (
          <Button
            onClick={handleTrain}
            disabled={
              !isModelConnected ||
              !isDatasetConnected ||
              !(connectedNodes.model?.data?.modelId && typeof connectedNodes.model.data.modelId === 'string' && connectedNodes.model.data.modelId.trim()) ||
              !(connectedNodes.dataset?.data?.datasetId && typeof connectedNodes.dataset.data.datasetId === 'string' && connectedNodes.dataset.data.datasetId.trim()) ||
              !outputSpace.trim() ||
              !textField.trim()
            }
            className="w-full"
            size="sm"
          >
            Train Another Model
          </Button>
        ) : (
          <Button
            onClick={handleTrain}
            disabled={
              !isModelConnected ||
              !isDatasetConnected ||
              !(connectedNodes.model?.data?.modelId && typeof connectedNodes.model.data.modelId === 'string' && connectedNodes.model.data.modelId.trim()) ||
              !(connectedNodes.dataset?.data?.datasetId && typeof connectedNodes.dataset.data.datasetId === 'string' && connectedNodes.dataset.data.datasetId.trim()) ||
              !outputSpace.trim() ||
              !textField.trim()
            }
            className="w-full"
            size="sm"
          >
            Train Model
          </Button>
        )}
      </div>

      {/* Bottom target handle - works exactly like top */}
      <Handle 
        type="target" 
        position={Position.Bottom}
        isConnectable={true}
        style={{ 
          background: '#3b82f6',
          width: '12px',
          height: '12px',
          zIndex: 10,
          pointerEvents: 'auto'
        }}
      />
    </div>
  );
}
