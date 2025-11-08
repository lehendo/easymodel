import  UnifiedPerplexityVisualization  from "./perplexity";
import SemanticDriftAnalysis from "./semantic";
import TokenEfficiencyVisualization from "./tokenefficiency";
import UnifiedGQSVisualization from "./gqs";


export function TextTab() {
  return (
    <div className="grid grid-cols-4 gap-4">
      <div className="col-span-2 row-span-1">
        <UnifiedPerplexityVisualization />
      </div>

      <div className="col-span-2 row-span-1">
        <UnifiedGQSVisualization/>
      </div>

      <div className="col-span-2 row-span-1">
        <SemanticDriftAnalysis />
      </div>

      <div className="col-span-2 row-span-1">
        <TokenEfficiencyVisualization />
      </div>
    </div>
  );
}