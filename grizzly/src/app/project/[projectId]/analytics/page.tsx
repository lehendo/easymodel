"use client";

import { useState } from "react";
import { Dock, DockIcon } from "../../../../components/ui/dock";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "../../../../components/ui/tooltip";
import { HomeIcon, ChartNoAxesColumn } from "lucide-react";
import Link from "next/link";
import { cn } from "../../../../lib/utils";
import { buttonVariants } from "../../../../components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "../../../../components/ui/card";
import LeftSidebar from "../../../../components/dashboard/LeftSidebar";
import RightSidebar from "../../../../components/dashboard/RightSidebar";
import { TextTab } from "../../../../components/analytics/tabtext";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../../../../components/ui/tabs";
import { ImageTab } from "../../../../components/analytics/tabimage";
import { usePathname, useParams } from "next/navigation";
import DockComponent from "../../../../components/dashboard/dock";
import { api } from "../../../../trpc/react";

export const dynamic = "force-dynamic";

const DashboardPage = () => {
  const [leftSidebarOpen, setLeftSidebarOpen] = useState(false);
  const [rightSidebarOpen, setRightSidebarOpen] = useState(false);
  const params = useParams();
  const projectId = params?.projectId as string;

  // Fetch analytics data
  const { data: analyticsData, isLoading, error } = api.analytics.getAnalyticsByProject.useQuery(
    { projectId },
    { enabled: !!projectId, refetchInterval: 5000 } // Refetch every 5 seconds to get latest data
  );

  // Parse analytics data
  let parsedAnalytics = null;
  let hasAnalyticsData = false;
  
  if (analyticsData?.data) {
    try {
      parsedAnalytics = typeof analyticsData.data === 'string' 
        ? JSON.parse(analyticsData.data) 
        : analyticsData.data;
      
      // Check if there's actual analytics data (not just empty object)
      hasAnalyticsData = !!(
        (parsedAnalytics?.perplexity?.epochs && parsedAnalytics.perplexity.epochs.length > 0) ||
        (parsedAnalytics?.perplexity?.training && parsedAnalytics.perplexity.training.length > 0) ||
        (parsedAnalytics?.perplexity?.validation && parsedAnalytics.perplexity.validation.length > 0) ||
        (parsedAnalytics?.semanticDrift?.segments && parsedAnalytics.semanticDrift.segments.length > 0) ||
        (parsedAnalytics?.gqs?.model && (
          (Array.isArray(parsedAnalytics.gqs.model) && parsedAnalytics.gqs.model.length > 0) ||
          (Array.isArray(parsedAnalytics.gqs.model[0]) && parsedAnalytics.gqs.model[0].length > 0)
        )) ||
        (parsedAnalytics?.gqs?.baseline && parsedAnalytics.gqs.baseline.length > 0) ||
        (parsedAnalytics?.tokenEfficiency?.pretrained && parsedAnalytics.tokenEfficiency.pretrained.length > 0) ||
        (parsedAnalytics?.tokenEfficiency?.finetuned && parsedAnalytics.tokenEfficiency.finetuned.length > 0)
      );
      
      console.log("[Analytics Page] Parsed analytics data:", {
        hasAnalyticsData,
        hasPerplexity: !!parsedAnalytics?.perplexity,
        perplexityEpochs: parsedAnalytics?.perplexity?.epochs?.length || 0,
        perplexityEpochsArray: parsedAnalytics?.perplexity?.epochs,
        perplexityTraining: parsedAnalytics?.perplexity?.training,
        perplexityValidation: parsedAnalytics?.perplexity?.validation,
        hasSemanticDrift: !!parsedAnalytics?.semanticDrift,
        semanticSegments: parsedAnalytics?.semanticDrift?.segments?.length || 0,
        semanticSegmentsArray: parsedAnalytics?.semanticDrift?.segments,
        semanticSimilarity: parsedAnalytics?.semanticDrift?.similarity,
        hasGQS: !!parsedAnalytics?.gqs,
        gqsModel: parsedAnalytics?.gqs?.model,
        gqsBaseline: parsedAnalytics?.gqs?.baseline,
        hasTokenEfficiency: !!parsedAnalytics?.tokenEfficiency,
        tokenEfficiencyFinetuned: parsedAnalytics?.tokenEfficiency?.finetuned,
        tokenEfficiencyPretrained: parsedAnalytics?.tokenEfficiency?.pretrained,
      });
    } catch (e) {
      console.error("Failed to parse analytics data:", e);
    }
  }

  return (
    <div className="flex h-screen flex-col lg:flex-row">
      {/* Left Sidebar */}
      <LeftSidebar />

      {/* Main Content */}

      <main className="relative h-full flex-1 overflow-y-auto bg-background p-4 sm:p-6 md:p-10">
        {/* <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-bold font-medium">
                Accuracy filler
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-1">
                <div className="text-sm font-medium">
                  Current Accuracy:{" "}
                  <span className="text-lg font-bold">87.5%</span>
                </div>
                <div className="text-sm font-medium">
                  Average Accuracy:{" "}
                  <span className="text-lg font-bold">85.2%</span>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-bold font-medium">
                Loss filler
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-1">
                <div className="text-sm font-medium">
                  Current Loss: <span className="text-lg font-bold">0.333</span>
                </div>
                <div className="text-sm font-medium">
                  Best Loss: <span className="text-lg font-bold">0.214</span>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-bold font-medium">
                Inference Latency filler
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-1">
                <div className="text-sm font-medium">
                  Average Latency:{" "}
                  <span className="text-lg font-bold">20ms</span>
                </div>
                <div className="text-sm font-medium">
                  P95 Latency: <span className="text-lg font-bold">54ms</span>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-bold font-medium">
                Memory Usage filler
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-1">
                <div className="text-sm font-medium">
                  Current Memory Usage:{" "}
                  <span className="text-lg font-bold">1.9GB</span>
                </div>
                <div className="text-sm font-medium">
                  Peak Memory Usage:{" "}
                  <span className="text-lg font-bold">9.6GB</span>
                </div>
              </div>
            </CardContent>
          </Card>
           </div> */}

        {/* Analytics Section */}

        {/* <Tabs defaultValue="text" className="mt-5">
          <TabsList>
            <TabsTrigger value="text">Text</TabsTrigger>
            <TabsTrigger value="image">Image</TabsTrigger>
            <TabsTrigger value="audio">Audio</TabsTrigger>
            <TabsTrigger value="video">Video</TabsTrigger>
            <TabsTrigger value="3d">3D</TabsTrigger>
            <TabsTrigger value="geospatial">Geospatial</TabsTrigger>
            <TabsTrigger value="tabular">Tabular</TabsTrigger>
            <TabsTrigger value="timeseries">Time-Series</TabsTrigger>
          </TabsList>

          <TabsContent value="text" className="space-y-4">
            <TextTab />
          </TabsContent>
          <TabsContent value="image" className="space-y-4">
            <ImageTab />
          </TabsContent>
            </Tabs> */}
        <DockComponent />
        <div className="h-14"></div>

        {/* Analytics Content */}
        {isLoading ? (
          <div className="flex h-full items-center justify-center">
            <div className="text-center">
              <div className="text-lg font-semibold">Loading analytics...</div>
              <div className="text-sm text-muted-foreground">Please wait</div>
            </div>
          </div>
        ) : error ? (
          <div className="flex h-full items-center justify-center">
            <div className="text-center">
              <div className="text-lg font-semibold text-red-600">Error loading analytics</div>
              <div className="text-sm text-muted-foreground">{error.message}</div>
            </div>
          </div>
        ) : !parsedAnalytics || !hasAnalyticsData ? (
          <div className="flex h-full items-center justify-center">
            <div className="text-center">
              <div className="text-lg font-semibold">No analytics data yet</div>
              <div className="text-sm text-muted-foreground">
                Please start a training job to view analytics.
              </div>
            </div>
          </div>
        ) : (
          <TextTab analyticsData={parsedAnalytics} />
        )}
      </main>

      {/* Right Sidebar */}
    </div>
  );
};

export default DashboardPage;
