"use client"

import { useRef, type ChangeEvent } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { X, Upload } from "lucide-react"
import type { ImageData } from "./green-coverage-analyzer"
import Image from "next/image"

interface ImageUploaderProps {
  images: ImageData[]
  onChange: (images: ImageData[]) => void
}

export default function ImageUploader({ images, onChange }: ImageUploaderProps) {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const currentYear = new Date().getFullYear()
  const years = Array.from({ length: 30 }, (_, i) => currentYear - i)

  const handleFileChange = async (e: ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files?.length) return

    const currentYear = new Date().getFullYear()

    const uploadImage = async (file: File) => {
      const formData = new FormData()
      formData.append("file", file)

      const res = await fetch("http://localhost:8000/upload", {
        method: "POST",
        body: formData,
      })
      const data = await res.json()

      return {
        id: Math.random().toString(36).substring(2, 9),
        file,
        year: currentYear,
        url: URL.createObjectURL(file),
        serverPath: data.file_path,
      }
    }

  const uploadedImages = await Promise.all(Array.from(e.target.files).map(uploadImage))
  onChange([...images, ...uploadedImages])

  if (fileInputRef.current) fileInputRef.current.value = ""
}

  const handleYearChange = (id: string, year: number) => {
    const updatedImages = images.map((img) => (img.id === id ? { ...img, year } : img))
    onChange(updatedImages)
  }

  const removeImage = (id: string) => {
    const updatedImages = images.filter((img) => img.id !== id)
    onChange(updatedImages)
  }

  return (
    <div className="space-y-6">
      <div
        className="flex flex-col items-center justify-center border-2 border-dashed border-gray-300 rounded-lg p-12 text-center hover:bg-gray-50 transition-colors cursor-pointer"
        onClick={() => fileInputRef.current?.click()}
      >
        <Upload className="h-10 w-10 text-gray-400 mb-2" />
        <div className="space-y-1">
          <p className="text-lg font-medium text-gray-700">Upload aerial images</p>
          <p className="text-sm text-gray-500">Drag and drop or click to browse</p>
          <p className="text-xs text-gray-400">Supports: JPG, PNG</p>
        </div>
        <Input
          ref={fileInputRef}
          type="file"
          accept="image/jpeg,image/png"
          multiple
          className="hidden"
          onChange={handleFileChange}
        />
      </div>

      {images.length > 0 && (
        <div className="space-y-4">
          <h3 className="text-lg font-medium">Uploaded Images ({images.length})</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {images.map((image) => (
              <Card key={image.id} className="overflow-hidden">
                <div className="relative aspect-video">
                  <Image
                    src={image.url || "/placeholder.svg"}
                    alt={`Aerial image from ${image.year}`}
                    fill
                    className="object-cover"
                  />
                  <Button
                    variant="destructive"
                    size="icon"
                    className="absolute top-2 right-2 h-8 w-8 rounded-full opacity-90"
                    onClick={() => removeImage(image.id)}
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </div>
                <CardContent className="p-4">
                  <div className="flex items-center gap-4">
                    <div className="flex-1">
                      <Label htmlFor={`year-${image.id}`} className="text-xs text-gray-500 mb-1 block">
                        Year
                      </Label>
                      <Select
                        value={image.year.toString()}
                        onValueChange={(value) => handleYearChange(image.id, Number.parseInt(value))}
                      >
                        <SelectTrigger id={`year-${image.id}`} className="w-full">
                          <SelectValue placeholder="Select year" />
                        </SelectTrigger>
                        <SelectContent>
                          {years.map((year) => (
                            <SelectItem key={year} value={year.toString()}>
                              {year}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="text-sm text-gray-500">
                      {image.file.name.length > 20 ? `${image.file.name.substring(0, 17)}...` : image.file.name}
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
