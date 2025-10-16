"use client"

import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Star, BookOpen, Users, Shield, Sparkles, Play, Check } from "lucide-react"
import Link from "next/link"

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* Header */}
      <header className="sticky top-0 z-50 w-full border-b border-border/40 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container flex h-16 items-center justify-between">
          <div className="flex items-center space-x-2">
            <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center magical-glow">
              <div className="w-4 h-4 bg-accent rounded-sm transform rotate-45"></div>
            </div>
            <span className="font-serif text-xl font-bold text-glow">Arcane Engine</span>
          </div>

          <nav className="hidden md:flex items-center space-x-6">
            <Link href="#features" className="text-sm font-medium hover:text-accent transition-colors">
              Features
            </Link>
            <Link href="#pricing" className="text-sm font-medium hover:text-accent transition-colors">
              Pricing
            </Link>
            <Link href="#community" className="text-sm font-medium hover:text-accent transition-colors">
              Community
            </Link>
            <Link href="#blog" className="text-sm font-medium hover:text-accent transition-colors">
              Blog
            </Link>
          </nav>

          <div className="flex items-center space-x-3">
            <Button
              variant="outline"
              size="sm"
              className="border-accent text-foreground hover:bg-accent hover:text-accent-foreground bg-transparent"
            >
              Sign In
            </Button>
            <Button size="sm" className="bg-accent text-accent-foreground hover:bg-accent/90 gold-glow">
              Start Free Adventure
            </Button>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="relative overflow-hidden particle-bg">
        <div className="absolute inset-0 bg-gradient-to-b from-primary/10 via-transparent to-transparent"></div>
        <div className="container relative py-24 md:py-32">
          <div className="mx-auto max-w-4xl text-center">
            <Badge variant="secondary" className="mb-6 bg-primary text-primary-foreground border-primary/50">
              <Sparkles className="w-3 h-3 mr-1" />
              Powered by Advanced AI
            </Badge>

            <h1 className="font-serif text-4xl md:text-6xl lg:text-7xl font-bold mb-6 text-glow">
              Unleash the Ultimate
              <br />
              <span className="text-foreground text-glow">Dungeon Master</span>
            </h1>

            <p className="text-xl md:text-2xl text-muted-foreground mb-8 max-w-2xl mx-auto leading-relaxed">
              Infinite stories. Unforgettable characters. Your D&D adventure begins now, powered by AI.
            </p>

            <div className="flex flex-col sm:flex-row gap-4 justify-center mb-8">
              <Link href="/story-select">
                <Button
                  size="lg"
                  className="bg-accent text-accent-foreground hover:bg-accent/90 gold-glow text-lg px-8 py-6"
                >
                  <Play className="w-5 h-5 mr-2" />
                  Start Your First Adventure
                </Button>
              </Link>
            </div>

            <p className="text-sm text-muted-foreground mb-4">No credit card required. Free trial available.</p>

            <div className="flex items-center justify-center gap-2 text-sm text-muted-foreground">
              <span>Trusted by adventurers from Faerûn to Eberron</span>
              <div className="flex gap-1">
                {[...Array(5)].map((_, i) => (
                  <Star key={i} className="w-4 h-4 fill-accent text-accent" />
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section className="py-24 bg-card/50">
        <div className="container">
          <div className="text-center mb-16">
            <h2 className="font-serif text-3xl md:text-4xl font-bold mb-4">Your Adventure in Three Simple Steps</h2>
          </div>

          <div className="grid md:grid-cols-3 gap-8 max-w-5xl mx-auto">
            <div className="text-center">
              <div className="w-16 h-16 bg-primary rounded-full flex items-center justify-center mx-auto mb-6 magical-glow">
                <Users className="w-8 h-8 text-primary-foreground" />
              </div>
              <div className="bg-accent text-accent-foreground rounded-full w-8 h-8 flex items-center justify-center mx-auto mb-4 text-sm font-bold">
                1
              </div>
              <h3 className="font-serif text-xl font-semibold mb-3">Create Your Hero</h3>
              <p className="text-muted-foreground">
                Describe your character in your own words. Our AI will guide you through a seamless, conversational
                creation process to bring your hero to life.
              </p>
            </div>

            <div className="text-center">
              <div className="w-16 h-16 bg-primary rounded-full flex items-center justify-center mx-auto mb-6 magical-glow">
                <BookOpen className="w-8 h-8 text-primary-foreground" />
              </div>
              <div className="bg-accent text-accent-foreground rounded-full w-8 h-8 flex items-center justify-center mx-auto mb-4 text-sm font-bold">
                2
              </div>
              <h3 className="font-serif text-xl font-semibold mb-3">Shape Your Story</h3>
              <p className="text-muted-foreground">
                Tell the AI what you want to do. Your choices, big or small, dynamically shape a unique narrative. No
                pre-written scripts, only pure freedom.
              </p>
            </div>

            <div className="text-center">
              <div className="w-16 h-16 bg-primary rounded-full flex items-center justify-center mx-auto mb-6 magical-glow">
                <Sparkles className="w-8 h-8 text-primary-foreground" />
              </div>
              <div className="bg-accent text-accent-foreground rounded-full w-8 h-8 flex items-center justify-center mx-auto mb-4 text-sm font-bold">
                3
              </div>
              <h3 className="font-serif text-xl font-semibold mb-3">Experience the Magic</h3>
              <p className="text-muted-foreground">
                Our AI handles all the rules, voices every character, and describes the world in vivid detail. You just
                focus on the adventure.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Features */}
      <section id="features" className="py-24">
        <div className="container">
          <div className="text-center mb-16">
            <h2 className="font-serif text-3xl md:text-4xl font-bold mb-4">
              More Than a Game Master. It's a Living World.
            </h2>
          </div>

          <div className="grid md:grid-cols-2 gap-8 max-w-4xl mx-auto">
            <Card className="border-primary/20 magical-glow">
              <CardContent className="p-6">
                <div className="w-12 h-12 bg-accent rounded-lg flex items-center justify-center mb-4">
                  <BookOpen className="w-6 h-6 text-accent-foreground" />
                </div>
                <h3 className="font-serif text-xl font-semibold mb-3">Infinite Storylines</h3>
                <p className="text-card-foreground">
                  Never play the same game twice. Our AI generates truly dynamic plots and side quests based on your
                  actions and backstory.
                </p>
              </CardContent>
            </Card>

            <Card className="border-primary/20 magical-glow">
              <CardContent className="p-6">
                <div className="w-12 h-12 bg-accent rounded-lg flex items-center justify-center mb-4">
                  <Users className="w-6 h-6 text-accent-foreground" />
                </div>
                <h3 className="font-serif text-xl font-semibold mb-3">Living NPCs</h3>
                <p className="text-card-foreground">
                  Converse with characters who have their own memories, motivations, and secrets. Your relationships
                  with them matter.
                </p>
              </CardContent>
            </Card>

            <Card className="border-primary/20 magical-glow">
              <CardContent className="p-6">
                <div className="w-12 h-12 bg-accent rounded-lg flex items-center justify-center mb-4">
                  <Shield className="w-6 h-6 text-accent-foreground" />
                </div>
                <h3 className="font-serif text-xl font-semibold mb-3">Smart Rules Engine</h3>
                <p className="text-card-foreground">
                  From complex combat to obscure skill checks, the AI acts as a fair and knowledgeable referee,
                  instantly applying the correct D&D 5e rules.
                </p>
              </CardContent>
            </Card>

            <Card className="border-primary/20 magical-glow">
              <CardContent className="p-6">
                <div className="w-12 h-12 bg-accent rounded-lg flex items-center justify-center mb-4">
                  <Sparkles className="w-6 h-6 text-accent-foreground" />
                </div>
                <h3 className="font-serif text-xl font-semibold mb-3">Stunning Visuals & Ambiance</h3>
                <p className="text-card-foreground">
                  Immerse yourself with AI-generated art for key locations, characters, and magical items, paired with
                  atmospheric sound effects.
                </p>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* Testimonials */}
      <section className="py-24 bg-card/50">
        <div className="container">
          <div className="text-center mb-16">
            <h2 className="font-serif text-3xl md:text-4xl font-bold mb-4">Hear from Your Fellow Adventurers</h2>
          </div>

          <div className="grid md:grid-cols-3 gap-8 max-w-6xl mx-auto">
            <Card className="border-primary/20">
              <CardContent className="p-6">
                <div className="flex items-center mb-4">
                  <div className="w-12 h-12 bg-primary rounded-full flex items-center justify-center mr-3">
                    <span className="text-primary-foreground font-bold">E</span>
                  </div>
                  <div>
                    <p className="font-semibold">Elara</p>
                    <p className="text-sm text-accent">First-Time Rogue</p>
                  </div>
                </div>
                <p className="text-card-foreground italic">
                  "I was always too intimidated to try D&D. Arcane Engine made it so easy and fun to get started! It's
                  like having a patient, creative DM available 24/7."
                </p>
              </CardContent>
            </Card>

            <Card className="border-primary/20">
              <CardContent className="p-6">
                <div className="flex items-center mb-4">
                  <div className="w-12 h-12 bg-primary rounded-full flex items-center justify-center mr-3">
                    <span className="text-primary-foreground font-bold">K</span>
                  </div>
                  <div>
                    <p className="font-semibold">Kael</p>
                    <p className="text-sm text-accent">Veteran Warlock</p>
                  </div>
                </div>
                <p className="text-card-foreground italic">
                  "The AI's ability to improvise is mind-blowing. It remembered a small detail from 5 sessions ago and
                  brought it back into the plot. I'm hooked."
                </p>
              </CardContent>
            </Card>

            <Card className="border-primary/20">
              <CardContent className="p-6">
                <div className="flex items-center mb-4">
                  <div className="w-12 h-12 bg-primary rounded-full flex items-center justify-center mr-3">
                    <span className="text-primary-foreground font-bold">A</span>
                  </div>
                  <div>
                    <p className="font-semibold">Aria</p>
                    <p className="text-sm text-accent">Dungeon Master</p>
                  </div>
                </div>
                <p className="text-card-foreground italic">
                  "Even as an experienced DM, I love using Arcane Engine for solo play. It's given me incredible
                  inspiration for my own campaigns."
                </p>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* Pricing */}
      <section id="pricing" className="py-24">
        <div className="container">
          <div className="text-center mb-16">
            <h2 className="font-serif text-3xl md:text-4xl font-bold mb-4">Choose Your Destiny</h2>
          </div>

          <div className="grid md:grid-cols-3 gap-8 max-w-5xl mx-auto">
            <Card className="border-border">
              <CardContent className="p-6">
                <h3 className="font-serif text-xl font-semibold mb-2">Free Trial</h3>
                <p className="text-3xl font-bold mb-4">$0</p>
                <p className="text-muted-foreground mb-6">Limited to one introductory adventure</p>
                <Button variant="outline" className="w-full bg-transparent">
                  Get Started
                </Button>
              </CardContent>
            </Card>

            <Card className="border-accent relative magical-glow">
              <div className="absolute -top-3 left-1/2 transform -translate-x-1/2">
                <Badge className="bg-accent text-accent-foreground">Most Popular</Badge>
              </div>
              <CardContent className="p-6">
                <h3 className="font-serif text-xl font-semibold mb-2">Adventurer</h3>
                <p className="text-3xl font-bold mb-1">
                  $19<span className="text-lg font-normal">/month</span>
                </p>
                <p className="text-muted-foreground mb-6">Unlimited adventures, core rulebooks, community features</p>
                <Button className="w-full bg-accent text-accent-foreground hover:bg-accent/90">Choose Plan</Button>
                <div className="mt-6 space-y-2">
                  <div className="flex items-center text-sm">
                    <Check className="w-4 h-4 text-accent mr-2" />
                    Unlimited adventures
                  </div>
                  <div className="flex items-center text-sm">
                    <Check className="w-4 h-4 text-accent mr-2" />
                    Core D&D 5e rulebooks
                  </div>
                  <div className="flex items-center text-sm">
                    <Check className="w-4 h-4 text-accent mr-2" />
                    Community features
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="border-border">
              <CardContent className="p-6">
                <h3 className="font-serif text-xl font-semibold mb-2">Loremaster</h3>
                <p className="text-3xl font-bold mb-1">
                  $39<span className="text-lg font-normal">/month</span>
                </p>
                <p className="text-muted-foreground mb-6">
                  All features plus expansion content and world creation tools
                </p>
                <Button variant="outline" className="w-full bg-transparent">
                  Choose Plan
                </Button>
                <div className="mt-6 space-y-2">
                  <div className="flex items-center text-sm">
                    <Check className="w-4 h-4 text-accent mr-2" />
                    Everything in Adventurer
                  </div>
                  <div className="flex items-center text-sm">
                    <Check className="w-4 h-4 text-accent mr-2" />
                    All expansion content
                  </div>
                  <div className="flex items-center text-sm">
                    <Check className="w-4 h-4 text-accent mr-2" />
                    Advanced AI model
                  </div>
                  <div className="flex items-center text-sm">
                    <Check className="w-4 h-4 text-accent mr-2" />
                    World creation tools
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* Final CTA */}
      <section className="py-24 bg-gradient-to-r from-primary/20 to-accent/20">
        <div className="container text-center">
          <h2 className="font-serif text-3xl md:text-4xl font-bold mb-6">Your Greatest Adventure is a Click Away</h2>
          <Link href="/story-select">
            <Button
              size="lg"
              className="bg-accent text-accent-foreground hover:bg-accent/90 gold-glow text-lg px-8 py-6"
            >
              <Play className="w-5 h-5 mr-2" />
              Start Your Free Adventure Today
            </Button>
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border bg-card/50 py-12">
        <div className="container">
          <div className="grid md:grid-cols-4 gap-8">
            <div>
              <div className="flex items-center space-x-2 mb-4">
                <div className="w-6 h-6 bg-primary rounded-lg flex items-center justify-center">
                  <div className="w-3 h-3 bg-accent rounded-sm transform rotate-45"></div>
                </div>
                <span className="font-serif text-lg font-bold">Arcane Engine</span>
              </div>
              <p className="text-sm text-muted-foreground">
                Unleashing infinite D&D adventures through the power of AI.
              </p>
            </div>

            <div>
              <h4 className="font-semibold mb-3">Product</h4>
              <div className="space-y-2 text-sm">
                <Link href="#features" className="block text-muted-foreground hover:text-accent">
                  Features
                </Link>
                <Link href="#pricing" className="block text-muted-foreground hover:text-accent">
                  Pricing
                </Link>
                <Link href="#" className="block text-muted-foreground hover:text-accent">
                  API
                </Link>
              </div>
            </div>

            <div>
              <h4 className="font-semibold mb-3">Community</h4>
              <div className="space-y-2 text-sm">
                <Link href="#" className="block text-muted-foreground hover:text-accent">
                  Discord
                </Link>
                <Link href="#" className="block text-muted-foreground hover:text-accent">
                  Reddit
                </Link>
                <Link href="#" className="block text-muted-foreground hover:text-accent">
                  Blog
                </Link>
              </div>
            </div>

            <div>
              <h4 className="font-semibold mb-3">Company</h4>
              <div className="space-y-2 text-sm">
                <Link href="#" className="block text-muted-foreground hover:text-accent">
                  About Us
                </Link>
                <Link href="#" className="block text-muted-foreground hover:text-accent">
                  Contact
                </Link>
                <Link href="#" className="block text-muted-foreground hover:text-accent">
                  Privacy Policy
                </Link>
              </div>
            </div>
          </div>

          <div className="border-t border-border mt-8 pt-8 flex flex-col md:flex-row justify-between items-center">
            <p className="text-sm text-muted-foreground">© 2024 Arcane Engine. All rights reserved.</p>
            <div className="flex space-x-4 mt-4 md:mt-0">
              <Link href="#" className="text-muted-foreground hover:text-accent">
                <span className="sr-only">Discord</span>
                <Users className="w-5 h-5" />
              </Link>
              <Link href="#" className="text-muted-foreground hover:text-accent">
                <span className="sr-only">Twitter</span>
                <BookOpen className="w-5 h-5" />
              </Link>
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}
