"use client";

import { create } from "zustand";
import {
  Node,
  Edge,
  Connection,
  NodeChange,
  EdgeChange,
  applyNodeChanges,
  applyEdgeChanges,
  addEdge,
} from "@xyflow/react";

const MIN_INACTIVITY = 60 * 1000; // 1 minute in milliseconds
const STORAGE_KEY = "reactflow-projects";

// Helper function to validate and fix node positions
const validateAndFixNodes = (nodes: any[]): Node[] => {
  if (!Array.isArray(nodes)) return [];
  return nodes.map((node: any) => {
    // Ensure node has required properties
    if (!node || !node.id) {
      // Skip invalid nodes
      return null;
    }
    if (!node.position || typeof node.position.x !== 'number' || typeof node.position.y !== 'number') {
      // Set default position if missing or invalid
      return {
        ...node,
        position: {
          x: node.position?.x || Math.random() * 400,
          y: node.position?.y || Math.random() * 400,
        },
      };
    }
    return node;
  }).filter((node): node is Node => node !== null); // Filter out null nodes
};

// Helper to persist to localStorage (SSR-safe)
const persistToLocalStorage = (projectId: string, nodes: Node[], edges: Edge[]) => {
  if (typeof window === 'undefined') return;
  
  try {
    const storage = localStorage.getItem(STORAGE_KEY);
    const projects = storage ? JSON.parse(storage) : {};
    projects[projectId] = { nodes, edges };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(projects));
  } catch (e) {
    console.error("Failed to persist to localStorage:", e);
  }
};

// Helper to load from localStorage (SSR-safe)
const loadFromLocalStorage = (projectId: string): { nodes: Node[]; edges: Edge[] } | null => {
  if (typeof window === 'undefined') return null;
  
  try {
    const storage = localStorage.getItem(STORAGE_KEY);
    if (!storage) return null;
    
    const projects = JSON.parse(storage);
    const project = projects[projectId];
    
    if (!project) return null;
    
    return {
      nodes: validateAndFixNodes(project.nodes || []),
      edges: project.edges || [],
    };
  } catch (e) {
    console.error("Failed to load from localStorage:", e);
    return null;
  }
};

// Singleton pattern for DB sync interval
let syncInterval: NodeJS.Timeout | null = null;
let syncToDBFn: ((projectId: string, nodes: Node[], edges: Edge[]) => Promise<void>) | null = null;

interface FlowStore {
  nodes: Node[];
  edges: Edge[];
  projectId: string | null;
  lastChangeAt: number;
  
  // Actions
  setProject: (projectId: string, loadFromDB?: (projectId: string) => Promise<{ nodes: any[]; edges?: Edge[] } | null>) => Promise<void>;
  onNodesChange: (changes: NodeChange[]) => void;
  onEdgesChange: (changes: EdgeChange[]) => void;
  onConnect: (connection: Connection) => void;
  addNode: (node: Node) => void;
  deleteNode: (nodeId: string) => void;
  updateEdges: (edges: Edge[]) => void;
  resetFlow: () => void;
  
  // Set DB sync function and schedule sync (singleton)
  setDBSync: (syncToDB: (projectId: string, nodes: Node[], edges: Edge[]) => Promise<void>) => void;
}

export const useFlowStore = create<FlowStore>((set, get) => ({
  nodes: [],
  edges: [],
  projectId: null,
  lastChangeAt: Date.now(),

  // Set DB sync function and schedule sync with singleton pattern
  setDBSync: (syncToDB) => {
    // Store the sync function
    syncToDBFn = syncToDB;
    
    // Prevent multiple intervals
    if (syncInterval) return;
    
    syncInterval = setInterval(() => {
      const state = get();
      if (
        state.projectId &&
        syncToDBFn &&
        Date.now() - state.lastChangeAt >= MIN_INACTIVITY
      ) {
        syncToDBFn(state.projectId, state.nodes, state.edges).catch((error) => {
          console.error("Failed to sync to database:", error);
        });
      }
    }, MIN_INACTIVITY);
  },

  // Set project - loads from localStorage or database (one-way hydration)
  setProject: async (projectId, loadFromDB) => {
    if (typeof window === 'undefined') return;
    
    // CRITICAL: Reset state cleanly before loading (prevents state bleed)
    set({
      projectId,
      nodes: [],
      edges: [],
      lastChangeAt: Date.now(),
    });
    
    // 1. Try localStorage first
    const local = loadFromLocalStorage(projectId);
    if (local) {
      set({
        projectId,
        nodes: local.nodes,
        edges: local.edges,
        lastChangeAt: Date.now(),
      });
      return;
    }
    
    // 2. Only if localStorage empty, try database (once)
    if (loadFromDB) {
      try {
        const dbData = await loadFromDB(projectId);
        if (dbData?.nodes) {
          const nodes = validateAndFixNodes(dbData.nodes);
          const edges = dbData.edges || [];
          set({
            projectId,
            nodes,
            edges,
            lastChangeAt: Date.now(),
          });
          // Save to localStorage
          persistToLocalStorage(projectId, nodes, edges);
        } else {
          // 3. Both empty - start blank (already reset above)
        }
      } catch (e) {
        console.error("Failed to load from database:", e);
        // Continue with empty state
      }
    }
  },

  // Handle node changes with atomic persistence
  onNodesChange: (changes) => {
    set((state) => {
      const nodes = applyNodeChanges(changes, state.nodes);
      
      // CRITICAL: Persist inside set callback (atomic)
      if (state.projectId && typeof window !== 'undefined') {
        persistToLocalStorage(state.projectId, nodes, state.edges);
      }
      
      return {
        nodes,
        lastChangeAt: Date.now(),
      };
    });
  },

  // Handle edge changes with atomic persistence
  onEdgesChange: (changes) => {
    set((state) => {
      const edges = applyEdgeChanges(changes, state.edges);
      
      // CRITICAL: Persist inside set callback (atomic)
      if (state.projectId && typeof window !== 'undefined') {
        persistToLocalStorage(state.projectId, state.nodes, edges);
      }
      
      return {
        edges,
        lastChangeAt: Date.now(),
      };
    });
  },

  // Add new edge
  onConnect: (connection) => {
    set((state) => {
      const edges = addEdge(connection, state.edges);
      
      // CRITICAL: Persist inside set callback (atomic)
      if (state.projectId && typeof window !== 'undefined') {
        persistToLocalStorage(state.projectId, state.nodes, edges);
      }
      
      return {
        edges,
        lastChangeAt: Date.now(),
      };
    });
  },

  // Helper: Add node (with atomic persistence)
  addNode: (node: Node) => {
    set((state) => {
      const nodes = [...state.nodes, node];
      
      // CRITICAL: Persist inside set callback (atomic)
      if (state.projectId && typeof window !== 'undefined') {
        persistToLocalStorage(state.projectId, nodes, state.edges);
      }
      
      return {
        nodes,
        lastChangeAt: Date.now(),
      };
    });
  },

  // Helper: Delete node and connected edges (with atomic persistence)
  deleteNode: (nodeId: string) => {
    set((state) => {
      const nodes = state.nodes.filter((node) => node.id !== nodeId);
      const edges = state.edges.filter(
        (edge) => edge.source !== nodeId && edge.target !== nodeId
      );
      
      // CRITICAL: Persist inside set callback (atomic)
      if (state.projectId && typeof window !== 'undefined') {
        persistToLocalStorage(state.projectId, nodes, edges);
      }
      
      return {
        nodes,
        edges,
        lastChangeAt: Date.now(),
      };
    });
  },

  // Helper: Update edges (for drag operations, with atomic persistence)
  updateEdges: (edges: Edge[]) => {
    set((state) => {
      // CRITICAL: Persist inside set callback (atomic)
      if (state.projectId && typeof window !== 'undefined') {
        persistToLocalStorage(state.projectId, state.nodes, edges);
      }
      
      return {
        edges,
        lastChangeAt: Date.now(),
      };
    });
  },

  // Reset flow state
  resetFlow: () => {
    set({
      nodes: [],
      edges: [],
      projectId: null,
      lastChangeAt: Date.now(),
    });
  },
}));
