"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { DocumentUpload } from "@/components/document-upload"
import { Input } from "@/components/ui/input"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Search, Database, FileText, Trash2 } from "lucide-react"
import { useToast } from "@/hooks/use-toast"

interface SearchResult {
  content: string
  metadata: {
    source: string
    type: string
    title?: string
    section?: string
  }
  similarity: number
}

export default function AdminPage() {
  const [searchQuery, setSearchQuery] = useState("")
  const [searchResults, setSearchResults] = useState<SearchResult[]>([])
  const [isSearching, setIsSearching] = useState(false)
  const [documentCount, setDocumentCount] = useState(0)
  const { toast } = useToast()

  const handleSearch = async () => {
    if (!searchQuery.trim()) return

    setIsSearching(true)
    try {
      const response = await fetch("/api/rag/search", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: searchQuery,
          limit: 10,
        }),
      })

      const data = await response.json()

      if (response.ok) {
        setSearchResults(data.results)
        toast({
          title: "Search Complete",
          description: `Found ${data.results.length} relevant results`,
        })
      } else {
        throw new Error(data.error)
      }
    } catch (error) {
      toast({
        title: "Search Failed",
        description: error instanceof Error ? error.message : "Failed to search documents",
        variant: "destructive",
      })
    } finally {
      setIsSearching(false)
    }
  }

  const clearDatabase = async () => {
    try {
      // This would need to be implemented as an API endpoint
      toast({
        title: "Clear Database",
        description: "This feature would clear all uploaded documents",
      })
    } catch (error) {
      toast({
        title: "Clear Failed",
        description: "Failed to clear database",
        variant: "destructive",
      })
    }
  }

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="container py-12">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-12">
            <h1 className="font-serif text-4xl font-bold mb-4">RAG Knowledge Base Admin</h1>
            <p className="text-xl text-muted-foreground">Manage your D&D documents and test the RAG system</p>
          </div>

          <Tabs defaultValue="upload" className="space-y-6">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="upload">Upload Documents</TabsTrigger>
              <TabsTrigger value="search">Test Search</TabsTrigger>
              <TabsTrigger value="manage">Manage Database</TabsTrigger>
            </TabsList>

            <TabsContent value="upload" className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <FileText className="h-5 w-5" />
                    Document Upload
                  </CardTitle>
                  <CardDescription>Upload D&D PDFs and text files to build your knowledge base</CardDescription>
                </CardHeader>
                <CardContent>
                  <DocumentUpload />
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="search" className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Search className="h-5 w-5" />
                    Test RAG Search
                  </CardTitle>
                  <CardDescription>Search your uploaded documents to test the RAG system</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex gap-3">
                    <Input
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      placeholder="Search for D&D content..."
                      className="flex-1"
                      onKeyDown={(e) => e.key === "Enter" && handleSearch()}
                    />
                    <Button onClick={handleSearch} disabled={isSearching || !searchQuery.trim()}>
                      {isSearching ? "Searching..." : "Search"}
                    </Button>
                  </div>

                  {searchResults.length > 0 && (
                    <div className="space-y-4">
                      <h3 className="font-semibold">Search Results:</h3>
                      {searchResults.map((result, index) => (
                        <Card key={index} className="border-l-4 border-l-primary">
                          <CardContent className="p-4">
                            <div className="flex items-start justify-between mb-2">
                              <div className="flex items-center gap-2">
                                <Badge variant="secondary">{result.metadata.type}</Badge>
                                <span className="text-sm text-muted-foreground">{result.metadata.source}</span>
                              </div>
                              <Badge variant="outline">{Math.round(result.similarity * 100)}% match</Badge>
                            </div>
                            <p className="text-sm leading-relaxed">{result.content}</p>
                          </CardContent>
                        </Card>
                      ))}
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="manage" className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Database className="h-5 w-5" />
                    Database Management
                  </CardTitle>
                  <CardDescription>Manage your document database and system settings</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <Card>
                      <CardContent className="p-4 text-center">
                        <div className="text-2xl font-bold text-primary mb-1">{documentCount}</div>
                        <div className="text-sm text-muted-foreground">Documents Stored</div>
                      </CardContent>
                    </Card>

                    <Card>
                      <CardContent className="p-4 text-center">
                        <div className="text-2xl font-bold text-primary mb-1">In-Memory</div>
                        <div className="text-sm text-muted-foreground">Storage Type</div>
                      </CardContent>
                    </Card>

                    <Card>
                      <CardContent className="p-4 text-center">
                        <div className="text-2xl font-bold text-primary mb-1">Active</div>
                        <div className="text-sm text-muted-foreground">RAG Status</div>
                      </CardContent>
                    </Card>
                  </div>

                  <div className="pt-4 border-t">
                    <Button variant="destructive" onClick={clearDatabase} className="flex items-center gap-2">
                      <Trash2 className="h-4 w-4" />
                      Clear All Documents
                    </Button>
                    <p className="text-sm text-muted-foreground mt-2">
                      This will remove all uploaded documents from the knowledge base.
                    </p>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </div>
  )
}
