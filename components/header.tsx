"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Menu, X } from "lucide-react"

export function Header() {
  const [isScrolled, setIsScrolled] = useState(false)
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false)

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 20)
    }
    window.addEventListener("scroll", handleScroll)
    return () => window.removeEventListener("scroll", handleScroll)
  }, [])

  const scrollToSection = (id: string) => {
    const element = document.getElementById(id)
    if (element) {
      element.scrollIntoView({ behavior: "smooth" })
      setIsMobileMenuOpen(false)
    }
  }

  return (
    <header
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        isScrolled ? "bg-card/95 backdrop-blur-md shadow-sm" : "bg-transparent"
      }`}
    >
      <div className="container mx-auto px-4 lg:px-8">
        <div className="flex items-center justify-between h-16 lg:h-20">
          {/* Logo */}
          <button
            onClick={() => scrollToSection("hero")}
            className="flex items-center hover:scale-105 transition-all duration-300"
          >
            <img
    src="/logo.png"
    alt="LMKR Logo"
    className="h-12 lg:h-16 w-auto"
  />
          </button>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex items-center gap-8">
            <button
              onClick={() => scrollToSection("about")}
              className="text-sm lg:text-base text-foreground/70 hover:text-foreground transition-all duration-300 hover:translate-y-[-2px]"
            >
              About
            </button>
            <button
              onClick={() => scrollToSection("services")}
              className="text-sm lg:text-base text-foreground/70 hover:text-foreground transition-all duration-300 hover:translate-y-[-2px]"
            >
              Services
            </button>
            <button
              onClick={() => scrollToSection("contact")}
              className="text-sm lg:text-base text-foreground/70 hover:text-foreground transition-all duration-300 hover:translate-y-[-2px]"
            >
              Contact
            </button>
            <Button
              onClick={() => scrollToSection("contact")}
              className="bg-secondary text-secondary-foreground hover:bg-secondary/90 hover:scale-105 transition-all duration-300"
            >
              Get Started
            </Button>
          </nav>

          {/* Mobile Menu Button */}
          <button
            onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
            className="md:hidden p-2 text-foreground transition-transform duration-300 hover:scale-110"
          >
            {isMobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
          </button>
        </div>

        {/* Mobile Navigation */}
        {isMobileMenuOpen && (
          <nav className="md:hidden py-4 space-y-4 animate-fade-in-up">
            <button
              onClick={() => scrollToSection("about")}
              className="block w-full text-left px-4 py-2 text-foreground/70 hover:text-foreground hover:bg-muted rounded-lg transition-all duration-300"
            >
              About
            </button>
            <button
              onClick={() => scrollToSection("services")}
              className="block w-full text-left px-4 py-2 text-foreground/70 hover:text-foreground hover:bg-muted rounded-lg transition-all duration-300"
            >
              Services
            </button>
            <button
              onClick={() => scrollToSection("contact")}
              className="block w-full text-left px-4 py-2 text-foreground/70 hover:text-foreground hover:bg-muted rounded-lg transition-all duration-300"
            >
              Contact
            </button>
            <div className="px-4">
              <Button
                onClick={() => scrollToSection("contact")}
                className="w-full bg-secondary text-secondary-foreground hover:bg-secondary/90 hover:scale-105 transition-all duration-300"
              >
                Get Started
              </Button>
            </div>
          </nav>
        )}
      </div>
    </header>
  )
}
