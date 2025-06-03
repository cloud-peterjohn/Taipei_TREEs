import type { ImageData } from "./green-coverage-analyzer"
import { ArrowUpRight, ArrowDownRight, Minus } from "lucide-react"

interface CoverageSummaryProps {
  images: ImageData[]
}

export default function CoverageSummary({ images }: CoverageSummaryProps) {
  // Filter images with coverage data and sort by year
  const validImages = [...images].filter((img) => img.coverage !== undefined).sort((a, b) => a.year - b.year)

  if (validImages.length === 0) {
    return <div className="text-gray-500">No data available for summary</div>
  }

  // Calculate summary statistics
  const coverageValues = validImages.map((img) => img.coverage || 0)
  const averageCoverage = coverageValues.reduce((sum, val) => sum + val, 0) / coverageValues.length

  const highestCoverage = Math.max(...coverageValues)
  const lowestCoverage = Math.min(...coverageValues)

  const highestYear = validImages.find((img) => img.coverage === highestCoverage)?.year
  const lowestYear = validImages.find((img) => img.coverage === lowestCoverage)?.year

  // Calculate overall trend
  const firstValue = coverageValues[0]
  const lastValue = coverageValues[coverageValues.length - 1]
  const overallChange = lastValue - firstValue
  const overallChangePercent = ((lastValue - firstValue) / firstValue) * 100

  // Generate insights
  let trendInsight = ""
  if (validImages.length >= 2) {
    if (Math.abs(overallChangePercent) < 5) {
      trendInsight = "Relatively stable green coverage over the observed period."
    } else if (overallChange > 0) {
      trendInsight = `Significant increase of ${overallChangePercent.toFixed(1)}% in green coverage from ${validImages[0].year} to ${validImages[validImages.length - 1].year}.`
    } else {
      trendInsight = `Significant decrease of ${Math.abs(overallChangePercent).toFixed(1)}% in green coverage from ${validImages[0].year} to ${validImages[validImages.length - 1].year}.`
    }
  }

  // Find largest year-to-year change
  let largestChange = 0
  let largestChangeYears = ""

  for (let i = 1; i < validImages.length; i++) {
    const prevCoverage = validImages[i - 1].coverage || 0
    const currCoverage = validImages[i].coverage || 0
    const change = Math.abs(currCoverage - prevCoverage)

    if (change > largestChange) {
      largestChange = change
      largestChangeYears = `${validImages[i - 1].year} to ${validImages[i].year}`
    }
  }

  return (
    <div className="space-y-6">
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="text-sm text-gray-500">Average Coverage</div>
          <div className="text-xl font-semibold">{averageCoverage.toFixed(1)}%</div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div className="p-3 bg-green-50 rounded-lg">
            <div className="text-xs text-gray-500 mb-1">Highest</div>
            <div className="text-lg font-medium text-green-700">{highestCoverage}%</div>
            <div className="text-xs text-gray-500">{highestYear}</div>
          </div>

          <div className="p-3 bg-red-50 rounded-lg">
            <div className="text-xs text-gray-500 mb-1">Lowest</div>
            <div className="text-lg font-medium text-red-700">{lowestCoverage}%</div>
            <div className="text-xs text-gray-500">{lowestYear}</div>
          </div>
        </div>
      </div>

      <div>
        <h4 className="text-sm font-medium mb-2">Overall Trend</h4>
        <div className="flex items-center gap-2 mb-2">
          {overallChange > 0 ? (
            <ArrowUpRight className="h-5 w-5 text-green-600" />
          ) : overallChange < 0 ? (
            <ArrowDownRight className="h-5 w-5 text-red-500" />
          ) : (
            <Minus className="h-5 w-5 text-gray-400" />
          )}

          <span
            className={`font-medium ${
              overallChange > 0 ? "text-green-600" : overallChange < 0 ? "text-red-500" : "text-gray-500"
            }`}
          >
            {overallChange > 0 ? "+" : ""}
            {overallChange.toFixed(1)}%
          </span>
        </div>
      </div>

      <div className="text-sm text-gray-700">
        <h4 className="font-medium mb-1">Insights:</h4>
        <ul className="space-y-2">
          <li>{trendInsight}</li>
          {largestChangeYears && (
            <li>
              Largest change observed from {largestChangeYears} ({largestChange.toFixed(1)}%).
            </li>
          )}
          {validImages.length < 3 && (
            <li className="text-amber-600">For more accurate trend analysis, consider adding more years of data.</li>
          )}
        </ul>
      </div>
    </div>
  )
}
