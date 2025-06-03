import { Suspense } from "react"
import GreenCoverageAnalyzer from "@/components/green-coverage-analyzer"
import { Skeleton } from "@/components/ui/skeleton"

export default function Home() {
  return (
    <main className="min-h-screen bg-gray-50 pb-16">
      <div className="container mx-auto px-4 py-8">
        <div className="flex justify-center mb-8">
          <img
            src="/logo.png"
            alt="Green Coverage Logo"
            className="h-32 w-auto"
          />
        </div>
        <h1 className="text-3xl font-bold text-center text-gray-800 mb-2">
          Green Coverage Visualization from Aerial Imagery
        </h1>
        <p className="text-center text-gray-600 mb-8">
          Upload aerial images to analyze vegetation coverage changes over time
        </p>

        <Suspense fallback={<Skeleton className="w-full h-[800px] rounded-lg" />}>
          <GreenCoverageAnalyzer />
        </Suspense>
      </div>
    </main>
  )
}
