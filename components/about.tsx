"use client"

import type React from "react"
import { useEffect, useRef } from "react"
import { Card } from "@/components/ui/card"
import { Building2, Users, Award, TrendingUp } from "lucide-react"

export function About() {
  const headerRef = useRef<HTMLDivElement>(null)
  const contentRef = useRef<HTMLDivElement>(null)
  const statsRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("revealed")
            observer.unobserve(entry.target)
          }
        })
      },
      { threshold: 0.1, rootMargin: "0px 0px -50px 0px" },
    )

    if (headerRef.current) observer.observe(headerRef.current)
    if (contentRef.current) observer.observe(contentRef.current)
    if (statsRef.current) observer.observe(statsRef.current)

    return () => observer.disconnect()
  }, [])

  const stats = [
    {
      icon: Building2,
      value: "500+",
      label: "Enterprise Clients",
    },
    {
      icon: Users,
      value: "1000+",
      label: "Technology Experts",
    },
    {
      icon: Award,
      value: "50+",
      label: "Industry Awards",
    },
    {
      icon: TrendingUp,
      value: "98%",
      label: "Client Satisfaction",
    },
  ]

  return (
    <section id="about" className="py-20 lg:py-32 bg-muted/30">
      <div className="container mx-auto px-4 lg:px-8">
        <div className="max-w-6xl mx-auto">
          {/* Section header */}
          <div ref={headerRef} className="text-center mb-16 space-y-4 scroll-reveal">
            <h2 className="text-3xl md:text-4xl lg:text-5xl font-bold text-balance text-foreground">
              About <span className="text-secondary">LMKR</span>
            </h2>
            <p className="text-lg md:text-xl text-muted-foreground max-w-3xl mx-auto text-pretty leading-relaxed">
              A global leader in enterprise technology solutions, delivering innovation and excellence for over two
              decades.
            </p>
          </div>

          {/* Main content */}
          <div ref={contentRef} className="grid lg:grid-cols-2 gap-12 items-center mb-16 scroll-reveal">
            <div className="space-y-6">
              <h3 className="text-2xl md:text-3xl font-bold text-foreground">Pioneering Digital Transformation</h3>
              <p className="text-base md:text-lg text-muted-foreground leading-relaxed">
                Since our founding, LMKR has been at the forefront of technological innovation, helping businesses
                navigate the complexities of digital transformation. Our commitment to excellence and customer success
                has made us a trusted partner for enterprises worldwide.
              </p>
              <p className="text-base md:text-lg text-muted-foreground leading-relaxed">
                We combine deep industry expertise with cutting-edge technology to deliver solutions that drive
                measurable business outcomes. Our team of certified professionals works collaboratively with clients to
                understand their unique challenges and craft tailored strategies for success.
              </p>
            </div>

            <div className="relative">
              <Card className="p-8 bg-card border-primary/20 shadow-lg">
                <div className="space-y-6">
                  <div className="flex items-start gap-4">
                    <div className="h-12 w-12 rounded-lg bg-primary/10 flex items-center justify-center flex-shrink-0">
                      <Sparkles className="h-6 w-6 text-primary" />
                    </div>
                    <div>
                      <h4 className="font-semibold text-lg mb-2 text-foreground">Our Mission</h4>
                      <p className="text-muted-foreground leading-relaxed">
                        To empower organizations with innovative technology solutions that drive growth, efficiency, and
                        competitive advantage.
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-4">
                    <div className="h-12 w-12 rounded-lg bg-secondary/10 flex items-center justify-center flex-shrink-0">
                      <Award className="h-6 w-6 text-secondary" />
                    </div>
                    <div>
                      <h4 className="font-semibold text-lg mb-2 text-foreground">Our Vision</h4>
                      <p className="text-muted-foreground leading-relaxed">
                        To be the most trusted technology partner, recognized for delivering excellence and creating
                        lasting value.
                      </p>
                    </div>
                  </div>
                </div>
              </Card>
            </div>
          </div>

          {/* Stats grid */}
          <div ref={statsRef} className="grid grid-cols-2 lg:grid-cols-4 gap-6 scroll-reveal">
            {stats.map((stat, index) => (
              <Card key={index} className="p-6 text-center hover:shadow-lg transition-shadow bg-card border-border">
                <div className="flex justify-center mb-4">
                  <div className="h-12 w-12 rounded-lg bg-primary/10 flex items-center justify-center">
                    <stat.icon className="h-6 w-6 text-primary" />
                  </div>
                </div>
                <div className="text-3xl font-bold text-foreground mb-2">{stat.value}</div>
                <div className="text-sm text-muted-foreground">{stat.label}</div>
              </Card>
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}

function Sparkles(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z" />
      <path d="M5 3v4" />
      <path d="M19 17v4" />
      <path d="M3 5h4" />
      <path d="M17 19h4" />
    </svg>
  )
}
