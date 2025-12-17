import  UnifiedPerplexityVisualization  from "./perplexity";
import SemanticDriftAnalysis from "./semantic";
import TokenEfficiencyVisualization from "./tokenefficiency";
import UnifiedGQSVisualization from "./gqs";

interface TextTabProps {
  analyticsData: any;
}

export function TextTab({ analyticsData }: TextTabProps) {
  // Debug logging
  if (analyticsData) {
    console.log("[TextTab] Analytics data received:", {
      hasPerplexity: !!analyticsData.perplexity,
      perplexityEpochs: analyticsData.perplexity?.epochs?.length || 0,
      perplexityEpochsArray: analyticsData.perplexity?.epochs,
      perplexityTraining: analyticsData.perplexity?.training,
      perplexityValidation: analyticsData.perplexity?.validation,
      hasSemanticDrift: !!analyticsData.semanticDrift,
      semanticSegments: analyticsData.semanticDrift?.segments?.length || 0,
      semanticSegmentsArray: analyticsData.semanticDrift?.segments,
      semanticSimilarity: analyticsData.semanticDrift?.similarity,
      hasGQS: !!analyticsData.gqs,
      gqsModel: analyticsData.gqs?.model,
      gqsBaseline: analyticsData.gqs?.baseline,
      hasTokenEfficiency: !!analyticsData.tokenEfficiency,
      tokenEfficiencyFinetuned: analyticsData.tokenEfficiency?.finetuned,
      tokenEfficiencyPretrained: analyticsData.tokenEfficiency?.pretrained,
    });
  }

  return (
    <div className="grid grid-cols-4 gap-4">
      <div className="col-span-2 row-span-1">
        <UnifiedPerplexityVisualization data={analyticsData?.perplexity} />
      </div>

      <div className="col-span-2 row-span-1">
        <UnifiedGQSVisualization data={analyticsData?.gqs} />
      </div>

      <div className="col-span-2 row-span-1">
        <SemanticDriftAnalysis data={analyticsData?.semanticDrift} />
      </div>

      <div className="col-span-2 row-span-1">
        <TokenEfficiencyVisualization data={analyticsData?.tokenEfficiency} />
      </div>
    </div>
  );
}