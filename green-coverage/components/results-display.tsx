import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import type { ImageData } from "./green-coverage-analyzer"
import Image from "next/image"

interface ResultsDisplayProps {
  images: ImageData[]
}

export default function ResultsDisplay({ images }: ResultsDisplayProps) {
  // Sort images by year
  const sortedImages = [...images].sort((a, b) => a.year - b.year)

  if (images.length === 0 || !images[0].coverage) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>No Results Available</CardTitle>
          <CardDescription>Please upload and analyze images to see results</CardDescription>
        </CardHeader>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Analysis Results</CardTitle>
        <CardDescription>Green coverage detection results for each uploaded image</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {sortedImages.map((image) => (
            <Card key={image.id} className="overflow-hidden">
              <Tabs defaultValue="original">
                <TabsList className="grid w-full grid-cols-2">
                  <TabsTrigger value="original">Original</TabsTrigger>
                  <TabsTrigger value="processed">Processed</TabsTrigger>
                </TabsList>
                <TabsContent value="original" className="m-0">
                  <div className="relative aspect-video">
                    <Image
                      src={image.url || "/placeholder.svg"}
                      alt={`Original aerial image from ${image.year}`}
                      fill
                      className="object-cover"
                    />
                  </div>
                </TabsContent>
                <TabsContent value="processed" className="m-0">
                  <div className="relative aspect-video">
                    <Image
                      src={image.processedUrl || image.url}
                      alt={`Processed aerial image from ${image.year}`}
                      fill
                      className="object-cover"
                    />
                  </div>
                </TabsContent>
              </Tabs>
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div className="font-medium">{image.year}</div>
                  <Badge className="bg-green-600 hover:bg-green-700">{image.coverage}% Green Coverage</Badge>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
