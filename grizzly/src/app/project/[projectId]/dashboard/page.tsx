"use client";

import { useState, useCallback, useRef, useEffect, useMemo } from "react";
import dynamic from "next/dynamic";
import {
  Background,
  Controls,
  addEdge,
  Node,
  Edge,
  Connection,
  ReactFlowProvider,
  useReactFlow,
  useStoreApi,
  useNodesState,
  useEdgesState,
} from "@xyflow/react";
import { usePathname } from "next/navigation"; // For detecting route changes
import { api } from "../../../../trpc/react";
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

const ReactFlow = dynamic(
  () => import("@xyflow/react").then((mod) => mod.ReactFlow),
  { ssr: false },
);

const MIN_INACTIVITY = 60 * 1000; // 1 minute in milliseconds

const NodeContextMenu = ({ children, onDelete }: { children: React.ReactNode; onDelete: () => void }) => (
  <ContextMenu>
    <ContextMenuTrigger>{children}</ContextMenuTrigger>
    <ContextMenuContent className="w-48">
      <ContextMenuItem onClick={onDelete}>Delete</ContextMenuItem>
    </ContextMenuContent>
  </ContextMenu>
);

const Home = () => {
  const pathname = usePathname();
  const projectId = pathname?.split("/")[2]; // Extract projectId from URL
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const [lastSyncTime, setLastSyncTime] = useState(Date.now());

  const reactFlowWrapper = useRef<HTMLDivElement | null>(null);
  const store = useStoreApi();
  const { getInternalNode } = useReactFlow();

  // Sync to DB
  const syncNodesMutation = api.project.updateNodes.useMutation();

  // Track if we've loaded nodes for this project to prevent clearing on re-render
  const [loadedProjectId, setLoadedProjectId] = useState<string | null>(null);

  // Load nodes/edges from localStorage when component mounts or projectId changes
  useEffect(() => {
    if (!projectId) return;
    
    // Only load if projectId changed (not on every render)
    if (loadedProjectId === projectId) return;
    
    const savedNodes = localStorage.getItem(`project-${projectId}-nodes`);
    const savedEdges = localStorage.getItem(`project-${projectId}-edges`);

    if (savedNodes) {
      try {
        const parsed = JSON.parse(savedNodes);
        if (Array.isArray(parsed)) {
          setNodes(parsed as Node[]);
        }
      } catch (e) {
        console.error("Failed to parse saved nodes:", e);
      }
    }
    if (savedEdges) {
      try {
        const parsed = JSON.parse(savedEdges);
        if (Array.isArray(parsed)) {
          setEdges(parsed);
        }
      } catch (e) {
        console.error("Failed to parse saved edges:", e);
      }
    }
    
    setLoadedProjectId(projectId);

    const syncInterval = setInterval(() => {
      if (Date.now() - lastSyncTime >= MIN_INACTIVITY) {
        syncNodesToDB();
      }
    }, MIN_INACTIVITY);

    return () => clearInterval(syncInterval); // Cleanup on unmount
  }, [projectId, loadedProjectId]); // Only reload when projectId actually changes

  // Sync the nodes and edges to localStorage on change (but don't clear if nodes become empty temporarily)
  useEffect(() => {
    if (projectId) {
      // Always save, even if empty, to preserve state
      localStorage.setItem(`project-${projectId}-nodes`, JSON.stringify(nodes));
      localStorage.setItem(`project-${projectId}-edges`, JSON.stringify(edges));
    }
  }, [nodes, edges, projectId]);

  // Sync nodes to the database
  const syncNodesToDB = async () => {
    if (nodes.length > 0 && projectId) {
      await syncNodesMutation.mutateAsync({ projectId, nodes });
      setLastSyncTime(Date.now()); // Update last sync time
    }
  };

  // Trigger DB sync when switching page (but don't clear localStorage)
  useEffect(() => {
    // Only sync if we have nodes and a valid projectId
    if (nodes.length > 0 && projectId) {
      syncNodesToDB();
    }
  }, [pathname]); // Only on pathname change, not on nodes change

  const onConnect = useCallback(
    (connection: Connection) => setEdges((eds) => addEdge(connection, eds)),
    [],
  );

  const onDeleteNode = useCallback(
    (nodeId: string) => {
      setNodes((nds) => nds.filter((node) => node.id !== nodeId));
      setEdges((eds) =>
        eds.filter((edge) => edge.source !== nodeId && edge.target !== nodeId),
      );
    },
    [setNodes, setEdges],
  );

  const nodeTypes = useMemo(
    () => ({
      huggingFace: (props) => (
        <NodeContextMenu onDelete={() => onDeleteNode(props.id)}>
          <HuggingFaceNode {...props} />
        </NodeContextMenu>
      ),
      dataset: (props) => (
        <NodeContextMenu onDelete={() => onDeleteNode(props.id)}>
          <HuggingFaceDatasetNode {...props} />
        </NodeContextMenu>
      ),
      finetuning: (props) => (
        <NodeContextMenu onDelete={() => onDeleteNode(props.id)}>
          <FinetuningNode {...props} />
        </NodeContextMenu>
      ),
    }),
    [onDeleteNode],
  );

  const getUpdatedConnection = useCallback(
    (node) => {
      const { nodeLookup } = store.getState();
      const internalNode = getInternalNode(node.id);
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
        { distance: Number.MAX_VALUE, node: null },
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
      };
    },
    [store, getInternalNode],
  );

  const onNodeDrag = useCallback(
    (_, node) => {
      const updatedConnection = getUpdatedConnection(node);
      setEdges((es) => {
        const nextEdges = es.filter((e) => e.className !== "temp");

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

        return nextEdges;
      });
    },
    [getUpdatedConnection, setEdges],
  );

  const onNodeDragStop = useCallback(
    (_, node) => {
      const updatedConnection = getUpdatedConnection(node);
      setEdges((es) => {
        const nextEdges = es.filter((e) => e.className !== "temp");

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

        return nextEdges;
      });
    },
    [getUpdatedConnection],
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

      setNodes((nds) => [...nds, newNode]);
    },
    [setNodes],
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
          colorMode="light"
          nodeTypes={nodeTypes}
          fitView={false}
          onNodeDrag={onNodeDrag}
          onNodeDragStop={onNodeDragStop}
          style={{ backgroundColor: "#F7F9FB" }}
        >
          {/* <DockComponent /> */}
          <Background color="#F7F9FB" />
          <Controls />
        </ReactFlow>
        <NodePalette onAddNode={(node) => setNodes((nds) => [...nds, node])} />
      </div>
    </div>
  );
};

export default () => (
  <ReactFlowProvider>
    <Home />
  </ReactFlowProvider>
);
