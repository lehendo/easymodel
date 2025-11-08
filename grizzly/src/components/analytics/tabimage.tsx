import  UnifiedPerplexityVisualization  from "@/components/analytics/perplexity";
import SemanticDriftAnalysis from "./semantic";
import TokenEfficiencyVisualization from "./tokenefficiency";
import UnifiedGQSVisualization from "@/components/analytics/gqs";
import { HeatmapOverlay } from "./imageanalytic";


export function ImageTab() {
  return (
    <div className="grid grid-cols-4 gap-4">
      <div className="col-span-2 row-span-1">
        <HeatmapOverlay />
      </div>

      <div className="col-span-2 row-span-1">
      </div>

      <div className="col-span-2 row-span-1">
      </div>

      <div className="col-span-2 row-span-1">
      </div>
    </div>
  );
}