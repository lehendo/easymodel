"use client";

import React, { useCallback, useRef, useEffect, useMemo } from "react";
import { useTheme } from "next-themes";
import nextDynamic from "next/dynamic";
import {
  Background,
  Controls,
  Node,
  Edge,
  Connection,
  ReactFlowProvider,
  useReactFlow,
  useStoreApi,
  type NodeProps,
} from "@xyflow/react";
import { useParams } from "next/navigation";
import { api } from "../../../../trpc/react";
import { useFlowStore } from "../../../../stores/flowStore";
import LeftSidebar from "../../../../components/dashboard/LeftSidebar";
import NodePalette from "../../../../components/react-flow/NodePalette";
import HuggingFaceDatasetNode from "../../../../components/react-flow/HuggingFaceDatasetNode";
import HuggingFaceNode from "../../../../components/react-flow/HuggingFaceNode";
import FinetuningNode from "../../../../components/react-flow/FinetuningNode";
import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuTrigger,
} from "../../../../components/ui/context-menu";
import DockComponent from "../../../../components/dashboard/dock";
import "@xyflow/react/dist/style.css";

export const dynamic = "force-dynamic";

const ReactFlow = nextDynamic(
  () => import("@xyflow/react").then((mod) => mod.ReactFlow),
  { ssr: false },
);

const NodeContextMenu = ({ children, onDelete }: { children: React.ReactNode; onDelete: () => void }) => (
  <ContextMenu>
    <ContextMenuTrigger>{children}</ContextMenuTrigger>
    <ContextMenuContent className="w-48">
      <ContextMenuItem onClick={onDelete}>Delete</ContextMenuItem>
    </ContextMenuContent>
  </ContextMenu>
);

const Home = () => {
  const params = useParams();
  const projectId = params?.projectId as string;
  const { theme } = useTheme();
  const [mounted, setMounted] = React.useState(false);

  // Avoid hydration mismatch
  useEffect(() => {
    setMounted(true);
  }, []);

  // Compute background color based on theme
  // Dark mode: hsl(240, 10%, 3.9%) ≈ #0a0a0f, Light mode: #F7F9FB
  const backgroundColor = useMemo(() => {
    if (!mounted) return "#F7F9FB"; // Default during SSR
    return theme === "dark" ? "#0a0a0f" : "#F7F9FB";
  }, [mounted, theme]);

  const reactFlowWrapper = useRef<HTMLDivElement | null>(null);
  const store = useStoreApi();
  const { getInternalNode } = useReactFlow();

  // Get Zustand store state and actions
  const { nodes, edges, setProject, onNodesChange, onEdgesChange, onConnect, setDBSync, addNode, deleteNode, updateEdges } = useFlowStore();

  // Sync to DB mutation
  const syncNodesMutation = api.project.updateNodes.useMutation();

  // Fetch project data from database (for initial load if localStorage is empty)
  const { data: projectData } = api.project.getProjectById.useQuery(
    { projectId: projectId || "" },
    { enabled: !!projectId }
  );

  // Set up DB sync function and schedule sync (ONCE per app lifecycle)
  useEffect(() => {
    const syncToDB = async (pid: string, nodes: Node[], edges: Edge[]) => {
      if (nodes.length > 0) {
        try {
          // Sync both nodes and edges to database
          await syncNodesMutation.mutateAsync({ 
            projectId: pid, 
            nodes,
            edges, // Include edges in sync
          });
        } catch (error) {
          console.error("Failed to sync nodes to database:", error);
        }
      }
    };

    // Set sync function and start timer (singleton pattern in store)
    // Called once per app lifecycle, not per projectId change
    setDBSync(syncToDB);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Empty deps - call once

  // Load project on mount or when projectId changes (NOT when projectData changes)
  useEffect(() => {
    if (!projectId) return;

    const loadFromDB = async (pid: string) => {
      // Access projectData from closure, but don't depend on it in effect deps
      // Store will decide whether to hydrate based on localStorage state
      const currentProjectData = projectData;
      if (!currentProjectData) return null;
      
      try {
        // Support both old format (just nodes array) and new format (nodes+edges object)
        let dbNodes: any[] = [];
        let dbEdges: Edge[] = [];
        
        // Type-safe access to projectData
        const data = currentProjectData as any;
        
        if (Array.isArray(data.nodes)) {
          // Old format: just nodes array
          dbNodes = data.nodes;
          dbEdges = [];
        } else if (data.nodes && typeof data.nodes === 'object' && 'nodes' in data.nodes) {
          // New format: { nodes: [], edges: [] } stored in nodes field
          dbNodes = data.nodes.nodes || [];
          dbEdges = data.nodes.edges || [];
        } else if ('edges' in data && Array.isArray(data.edges)) {
          // Edges at top level (from getProjectById parsing)
          dbNodes = Array.isArray(data.nodes) ? data.nodes : [];
          dbEdges = data.edges;
        } else {
          // Fallback
          dbNodes = Array.isArray(data.nodes) ? data.nodes : [];
          dbEdges = [];
        }
        
        if (dbNodes.length > 0 || dbEdges.length > 0) {
          return {
            nodes: dbNodes,
            edges: dbEdges,
          };
        }
      } catch (e) {
        console.error("Failed to parse database nodes:", e);
      }
      return null;
    };

    setProject(projectId, loadFromDB);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId]); // ONLY depend on projectId - store handles hydration logic

  const onDeleteNode = useCallback(
    (nodeId: string) => {
      deleteNode(nodeId);
    },
    [deleteNode],
  );

  const nodeTypes = useMemo(
    () => ({
      huggingFace: (props: NodeProps) => (
        <NodeContextMenu onDelete={() => onDeleteNode(props.id)}>
          <HuggingFaceNode {...props} />
        </NodeContextMenu>
      ),
      dataset: (props: NodeProps) => (
        <NodeContextMenu onDelete={() => onDeleteNode(props.id)}>
          <HuggingFaceDatasetNode {...props} />
        </NodeContextMenu>
      ),
      finetuning: (props: NodeProps) => (
        <NodeContextMenu onDelete={() => onDeleteNode(props.id)}>
          <FinetuningNode {...props} />
        </NodeContextMenu>
      ),
    }),
    [onDeleteNode],
  );

  const getUpdatedConnection = useCallback(
    (node: Node) => {
      const { nodeLookup } = store.getState();
      const internalNode = getInternalNode(node.id);
      if (!internalNode) return null;
      
      const MIN_DISTANCE = 150;

      const closestNode = Array.from(nodeLookup.values()).reduce(
        (res, n) => {
          if (n.id !== internalNode.id) {
            const dx =
              n.internals.positionAbsolute.x -
              internalNode.internals.positionAbsolute.x;
            const dy =
              n.internals.positionAbsolute.y -
              internalNode.internals.positionAbsolute.y;
            const d = Math.sqrt(dx * dx + dy * dy);

            if (d < res.distance && d < MIN_DISTANCE) {
              res.distance = d;
              res.node = n;
            }
          }
          return res;
        },
        { distance: Number.MAX_VALUE, node: null as any },
      );

      if (!closestNode.node) return null;

      const closeNodeIsSource =
        closestNode.node.internals.positionAbsolute.y <
        internalNode.internals.positionAbsolute.y;

      return {
        id: closeNodeIsSource
          ? `${closestNode.node.id}-${node.id}`
          : `${node.id}-${closestNode.node.id}`,
        source: closeNodeIsSource ? closestNode.node.id : node.id,
        target: closeNodeIsSource ? node.id : closestNode.node.id,
      } as any;
    },
    [store, getInternalNode],
  );

  const onNodeDrag = useCallback(
    (_: any, node: Node) => {
      const updatedConnection = getUpdatedConnection(node);
      const state = useFlowStore.getState();
      const nextEdges = state.edges.filter((e) => e.className !== "temp");

      if (
        updatedConnection &&
        !nextEdges.find(
          (ne) =>
            ne.source === updatedConnection.source &&
            ne.target === updatedConnection.target,
        )
      ) {
        updatedConnection.className = "temp";
        nextEdges.push(updatedConnection);
      }

      updateEdges(nextEdges);
    },
    [getUpdatedConnection, updateEdges],
  );

  const onNodeDragStop = useCallback(
    (_: any, node: Node) => {
      const updatedConnection = getUpdatedConnection(node);
      const state = useFlowStore.getState();
      const nextEdges = state.edges.filter((e) => e.className !== "temp");

      if (
        updatedConnection &&
        !nextEdges.find(
          (ne) =>
            ne.source === updatedConnection.source &&
            ne.target === updatedConnection.target,
        )
      ) {
        nextEdges.push(updatedConnection);
      }

      updateEdges(nextEdges);
    },
    [getUpdatedConnection, updateEdges],
  );

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();

      if (!reactFlowWrapper.current) return;

      const reactFlowBounds = reactFlowWrapper.current.getBoundingClientRect();
      const type = event.dataTransfer.getData("application/reactflow");

      if (!type) {
        console.error("No node type provided in drag data.");
        return;
      }

      const position = {
        x: event.clientX - reactFlowBounds.left,
        y: event.clientY - reactFlowBounds.top,
      };

      const newNode: Node = {
        id: `${type}-${Date.now()}`,
        type,
        position,
        data: { label: `${type} Node` },
      };

      // Add node using store (which handles persistence atomically)
      addNode(newNode);
    },
    [addNode],
  );

  return (
    <div className="flex h-screen">
      <LeftSidebar />
      <div className="relative flex-1" ref={reactFlowWrapper}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          onDragOver={(e) => e.preventDefault()}
          onDrop={onDrop}
          colorMode={mounted && theme === "dark" ? "dark" : "light"}
          nodeTypes={nodeTypes}
          fitView={false}
          onNodeDrag={onNodeDrag}
          onNodeDragStop={onNodeDragStop}
          className="bg-background"
        >
          <DockComponent />
          <Background color={backgroundColor} />
          <Controls />
        </ReactFlow>
        <NodePalette 
          onAddNode={(node: any) => addNode(node)}
          getReactFlowBounds={() => reactFlowWrapper.current?.getBoundingClientRect()} 
        />
      </div>
    </div>
  );
};

export default () => (
  <ReactFlowProvider>
    <Home />
  </ReactFlowProvider>
);
