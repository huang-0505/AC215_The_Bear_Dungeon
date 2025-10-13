import LandingPage from "@/components/landing-page"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Users, User } from "lucide-react"

export default function HomePage() {
  return (
    <div>
      <LandingPage />

      <div className="fixed bottom-6 right-6 z-50 flex flex-col gap-3">
        <Link href="/multiplayer">
          <Button className="bg-accent text-accent-foreground hover:bg-accent/90 magical-glow flex items-center gap-2">
            <Users className="w-4 h-4" />
            Multiplayer
          </Button>
        </Link>

        <Link href="/story-select">
          <Button className="bg-primary text-primary-foreground hover:bg-primary/90 magical-glow flex items-center gap-2">
            <User className="w-4 h-4" />
            Solo Adventure
          </Button>
        </Link>
      </div>
    </div>
  )
}
