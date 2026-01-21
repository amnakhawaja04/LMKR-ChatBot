"use client"

import { Button } from "@/components/ui/button"
import { ArrowRight, Sparkles } from "lucide-react"

export function Hero() {
  const scrollToSection = (id: string) => {
    const element = document.getElementById(id)
    if (element) {
      element.scrollIntoView({ behavior: "smooth" })
    }
  }

  return (
    <section
      id="hero"
      className="min-h-screen bg-gradient-to-br from-[#0a1a2f] via-[#071423] to-[#020b18]"
    >
      {/* Decorative elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-64 h-64 bg-primary/5 rounded-full blur-3xl" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-secondary/5 rounded-full blur-3xl" />
      </div>

      <div className="container mx-auto px-4 lg:px-8 py-20 lg:py-32 relative z-10">
        <div className="max-w-4xl mx-auto text-center space-y-8 animate-fade-in-up">
          {/* Badge */}
          <div className="inline-flex items-center gap-2 px-4 py-2 bg-primary/10 border border-primary/20 rounded-full text-sm text-primary animate-fade-in-up">
            <Sparkles size={16} />
            <span className="font-medium">Enterprise Technology Solutions</span>
          </div>

          {/* Main heading */}
          <h1 className="text-4xl md:text-5xl lg:text-7xl font-bold text-balance animate-fade-in-up animate-delay-100">
            Transforming Business Through <span className="text-primary">Innovation</span> and{" "}
            <span className="text-secondary">Technology</span>
          </h1>

          {/* Subheading */}
          <p className="text-lg md:text-xl lg:text-2xl text-muted-foreground max-w-3xl mx-auto text-pretty leading-relaxed animate-fade-in-up animate-delay-200">
            LMKR delivers cutting-edge software solutions that empower enterprises to achieve digital excellence and
            drive sustainable growth.
          </p>

          {/* CTAs */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 pt-4 animate-fade-in-up animate-delay-300">
            <Button
              onClick={() => scrollToSection("contact")}
              size="lg"
              className="bg-primary text-primary-foreground hover:bg-primary/90 hover:scale-105 text-base px-8 h-12 group transition-all duration-300"
            >
              Get Started
              <ArrowRight className="ml-2 h-5 w-5 group-hover:translate-x-1 transition-transform duration-300" />
            </Button>
            <Button
              onClick={() => scrollToSection("services")}
              size="lg"
              variant="outline"
              className="border-primary text-primary hover:bg-primary hover:text-primary-foreground hover:scale-105 text-base px-8 h-12 transition-all duration-300"
            >
              Explore Services
            </Button>
          </div>

          {/* Trust indicators */}
          <div className="pt-12 flex flex-wrap items-center justify-center gap-8 text-sm text-muted-foreground animate-fade-in-up animate-delay-400">
            <div className="flex items-center gap-2">
              <div className="h-2 w-2 bg-secondary rounded-full" />
              <span>Trusted by 500+ Enterprises</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="h-2 w-2 bg-primary rounded-full" />
              <span>20+ Years of Excellence</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="h-2 w-2 bg-secondary rounded-full" />
              <span>Global Presence</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
