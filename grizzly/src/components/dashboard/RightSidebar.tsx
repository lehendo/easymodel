"use client";

import { TextSearch, Rocket, Loader } from "lucide-react";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Switch } from "@/components/ui/switch";
import {
  AlertDialog,
  AlertDialogTrigger,
  AlertDialogContent,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogDescription,
  AlertDialogFooter,
} from "@/components/ui/alert-dialog";

export default function RightSidebar() {
  const [isLaunching, setIsLaunching] = useState(false);
  const [showDialog, setShowDialog] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false); // Controls sidebar state

  const handleLaunch = () => {
    setIsLaunching(true);
    setTimeout(() => {
      setIsLaunching(false);
      setShowDialog(true); // Show dialog after loading
    }, 4000); // Spinner lasts for 10 seconds
  };

  const handleClose = () => {
    setShowDialog(false);
    setIsSidebarOpen(false); // Close sidebar when dialog is dismissed
  };

  return (
    <Sheet open={isSidebarOpen} onOpenChange={setIsSidebarOpen}>
      <SheetTrigger asChild>
        <Button
          className="fixed right-[-24px] top-1/2 z-10 -translate-y-1/2 rotate-90"
          variant="destructive"
        >
          Deploy
        </Button>
      </SheetTrigger>
      <SheetContent side="right">
        <div className="py-4">
          {/* Deployment Options */}
          <Card>
            <CardHeader>
              <CardTitle>Deployment Options</CardTitle>
              <CardDescription>
                Tune hyperparameters and dataset specifics here to create your
                perfect model!
              </CardDescription>
            </CardHeader>
            <CardContent className="grid gap-4">
              <div className="flex items-center space-x-4 rounded-md border p-4">
                <TextSearch />
                <div className="flex-1 space-y-1">
                  <p className="text-sm font-medium leading-none">
                    Infer Data Columns
                  </p>
                  <p className="text-sm text-muted-foreground">
                    When selected, we will optimally choose how to parse the
                    data used during training.
                  </p>
                </div>
                <Switch />
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Launch Button */}
        <div className="absolute bottom-4 left-4 right-4">
          <Button
            className="w-full bg-orange-500 py-4 text-white hover:bg-orange-600"
            onClick={handleLaunch}
            disabled={isLaunching}
          >
            {isLaunching ? (
              <Loader className="mr-2 animate-spin" />
            ) : (
              <Rocket className="mr-2" />
            )}
            {isLaunching ? "Launching..." : "Launch!"}
          </Button>
        </div>

        {/* Alert Dialog */}
        {showDialog && (
          <AlertDialog open={showDialog} onOpenChange={setShowDialog}>
            <AlertDialogContent className="animate-fade-in">
              {" "}
              {/* Smooth fade-in */}
              <AlertDialogHeader>
                <AlertDialogTitle>Launch Complete</AlertDialogTitle>
                <AlertDialogDescription>
                  Your model has been successfully deployed!
                </AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <Button onClick={handleClose}>OK</Button> {/* Closes sidebar */}
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
        )}
      </SheetContent>
    </Sheet>
  );
}
