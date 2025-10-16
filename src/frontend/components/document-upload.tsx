"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Upload, FileText, Loader2 } from "lucide-react"
import { useToast } from "@/hooks/use-toast"

export function DocumentUpload() {
  const [file, setFile] = useState<File | null>(null)
  const [type, setType] = useState<string>("")
  const [title, setTitle] = useState<string>("")
  const [isUploading, setIsUploading] = useState(false)
  const { toast } = useToast()

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0]
    if (selectedFile) {
      setFile(selectedFile)
      if (!title) {
        setTitle(selectedFile.name.replace(/\.[^/.]+$/, ""))
      }
    }
  }

  const handleUpload = async () => {
    if (!file || !type) {
      toast({
        title: "Missing Information",
        description: "Please select a file and document type.",
        variant: "destructive",
      })
      return
    }

    setIsUploading(true)

    try {
      const formData = new FormData()
      formData.append("file", file)
      formData.append("type", type)
      formData.append("title", title)

      const response = await fetch("/api/rag/upload", {
        method: "POST",
        body: formData,
      })

      const result = await response.json()

      if (response.ok) {
        toast({
          title: "Upload Successful",
          description: result.message,
        })

        // Reset form
        setFile(null)
        setType("")
        setTitle("")
        const fileInput = document.getElementById("file-upload") as HTMLInputElement
        if (fileInput) fileInput.value = ""
      } else {
        throw new Error(result.error)
      }
    } catch (error) {
      toast({
        title: "Upload Failed",
        description: error instanceof Error ? error.message : "Failed to upload document",
        variant: "destructive",
      })
    } finally {
      setIsUploading(false)
    }
  }

  return (
    <Card className="w-full max-w-md">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <FileText className="h-5 w-5" />
          Upload D&D Documents
        </CardTitle>
        <CardDescription>Upload PDFs or text files to build your knowledge base</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="file-upload">Document File</Label>
          <Input id="file-upload" type="file" accept=".pdf,.txt" onChange={handleFileChange} disabled={isUploading} />
        </div>

        <div className="space-y-2">
          <Label htmlFor="document-type">Document Type</Label>
          <Select value={type} onValueChange={setType} disabled={isUploading}>
            <SelectTrigger>
              <SelectValue placeholder="Select document type" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="rule">Rules & Mechanics</SelectItem>
              <SelectItem value="lore">Lore & Worldbuilding</SelectItem>
              <SelectItem value="monster">Monsters & NPCs</SelectItem>
              <SelectItem value="spell">Spells & Magic</SelectItem>
              <SelectItem value="item">Items & Equipment</SelectItem>
              <SelectItem value="campaign">Campaign & Adventures</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-2">
          <Label htmlFor="document-title">Title (Optional)</Label>
          <Input
            id="document-title"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder="Document title"
            disabled={isUploading}
          />
        </div>

        <Button onClick={handleUpload} disabled={!file || !type || isUploading} className="w-full">
          {isUploading ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Processing...
            </>
          ) : (
            <>
              <Upload className="mr-2 h-4 w-4" />
              Upload Document
            </>
          )}
        </Button>
      </CardContent>
    </Card>
  )
}
