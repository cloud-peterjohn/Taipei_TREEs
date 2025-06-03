"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Loader2, Info } from "lucide-react"
import ImageUploader from "@/components/image-uploader"
import ResultsDisplay from "@/components/results-display"
import CoverageChart from "@/components/coverage-chart"
import CoverageSummary from "@/components/coverage-summary"

export type ImageData = {
  id: string
  file: File
  year: number
  url: string
  processedUrl?: string
  coverage?: number
  serverPath?: string
}

export default function GreenCoverageAnalyzer() {
  const [images, setImages] = useState<ImageData[]>([])
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [isAnalyzed, setIsAnalyzed] = useState(false)
  const [activeTab, setActiveTab] = useState("upload")

  const handleImagesChange = (newImages: ImageData[]) => {
    setImages(newImages)
    setIsAnalyzed(false)
  }

  const runAnalysis = async () => {
    if (images.length === 0) return
    setIsAnalyzing(true)

    try {
      const response = await fetch("http://localhost:8000/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(
          images.map((img) => ({
            id: img.id,
            year: img.year,
            file_path: img.serverPath,
          }))
        ),
      })

      const result = await response.json()
      console.log("Result from backend:", result) 

      const analyzedImages = images.map((img) => {
        const processed = result.find((r: any) => r.id === img.id)
        console.log(processed?.coverage, processed?.processed_url)
        return {
          ...img,
          coverage: processed?.coverage,
          processedUrl: processed?.processed_url,
        }
      })

      setImages(analyzedImages)
      setIsAnalyzing(false)
      setIsAnalyzed(true)
      setActiveTab("results")
    } catch (error) {
      console.error("分析錯誤：", error)
      setIsAnalyzing(false)
    }
  }

  return (
    <div className="space-y-6">
      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="upload">Upload Images</TabsTrigger>
          <TabsTrigger value="results" disabled={!isAnalyzed}>
            Results
          </TabsTrigger>
          <TabsTrigger value="visualization" disabled={!isAnalyzed}>
            Visualization
          </TabsTrigger>
        </TabsList>

        <TabsContent value="upload" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle>Upload Aerial Images</CardTitle>
              <CardDescription>Upload multiple aerial images and assign a year to each one</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <ImageUploader images={images} onChange={handleImagesChange} />

              {images.length > 0 && (
                <div className="flex justify-center mt-6">
                  <Button
                    size="lg"
                    onClick={runAnalysis}
                    disabled={isAnalyzing || images.length === 0}
                    className="bg-green-600 hover:bg-green-700"
                  >
                    {isAnalyzing ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Analyzing green areas...
                      </>
                    ) : (
                      <>Run Green Coverage Analysis</>
                    )}
                  </Button>
                </div>
              )}

              {images.length === 0 && (
                <Alert>
                  <Info className="h-4 w-4" />
                  <AlertDescription>Please upload at least one aerial image to begin analysis.</AlertDescription>
                </Alert>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="results" className="mt-6">
          <ResultsDisplay images={images} />
        </TabsContent>

        <TabsContent value="visualization" className="mt-6">
          <div className="grid gap-6 md:grid-cols-3">
            <Card className="md:col-span-2">
              <CardHeader>
                <CardTitle>Green Coverage Over Time</CardTitle>
                <CardDescription>Visualization of vegetation coverage changes across years</CardDescription>
              </CardHeader>
              <CardContent>
                <CoverageChart images={images} />
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Summary</CardTitle>
                <CardDescription>Key insights from your analysis</CardDescription>
              </CardHeader>
              <CardContent>
                <CoverageSummary images={images} />
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>

      {isAnalyzed && activeTab === "upload" && (
        <Alert className="bg-green-50 border-green-200">
          <Info className="h-4 w-4 text-green-600" />
          <AlertDescription className="text-green-800">
            Analysis complete! View the results and visualization tabs to see your data.
          </AlertDescription>
        </Alert>
      )}
    </div>
  )
}
