"use client";

// Force dynamic rendering
export const dynamic = 'force-dynamic';

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
import { usePathname } from "next/navigation";
import DockComponent from "../../../../components/dashboard/dock";

const DashboardPage = () => {
  const [leftSidebarOpen, setLeftSidebarOpen] = useState(false);
  const [rightSidebarOpen, setRightSidebarOpen] = useState(false);

  return (
    <div className="flex h-screen flex-col lg:flex-row">
      {/* Left Sidebar */}
      <LeftSidebar />

      {/* Main Content */}

      <main className="relative h-full flex-1 overflow-y-auto bg-gray-100 p-4 sm:p-6 md:p-10">
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
        <TextTab />

        {/* <div className="mt-4">
        <TextTab />
        </div> */}
      </main>

      {/* Right Sidebar */}
    </div>
  );
};

export default DashboardPage;
