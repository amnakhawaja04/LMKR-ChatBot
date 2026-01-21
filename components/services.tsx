"use client"

import { useEffect, useRef } from "react"
import { Card } from "@/components/ui/card"
import { Code2, Cloud, Shield, Workflow, Database, LineChart } from "lucide-react"

export function Services() {
  const headerRef = useRef<HTMLDivElement>(null)
  const gridRef = useRef<HTMLDivElement>(null)

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
    if (gridRef.current) observer.observe(gridRef.current)

    return () => observer.disconnect()
  }, [])

  const services = [
    {
      icon: Code2,
      title: "Software Development",
      description:
        "Custom software solutions built with modern technologies to meet your specific business requirements and scale with your growth.",
    },
    {
      icon: Cloud,
      title: "Cloud Solutions",
      description:
        "Comprehensive cloud architecture, migration, and optimization services for AWS, Azure, and Google Cloud platforms.",
    },
    {
      icon: Shield,
      title: "Cybersecurity",
      description:
        "Enterprise-grade security solutions to protect your digital assets, ensure compliance, and mitigate cyber threats.",
    },
    {
      icon: Workflow,
      title: "Digital Transformation",
      description:
        "Strategic consulting and implementation services to modernize your business processes and technology infrastructure.",
    },
    {
      icon: Database,
      title: "Data & Analytics",
      description:
        "Advanced data management, business intelligence, and AI-powered analytics to drive informed decision-making.",
    },
    {
      icon: LineChart,
      title: "Enterprise Integration",
      description:
        "Seamless integration of enterprise systems, APIs, and third-party applications to optimize your technology ecosystem.",
    },
  ]

  return (
    <section id="services" className="py-20 lg:py-32 bg-background">
      <div className="container mx-auto px-4 lg:px-8">
        <div className="max-w-6xl mx-auto">
          <div ref={headerRef} className="text-center mb-16 space-y-4 scroll-reveal">
            <h2 className="text-3xl md:text-4xl lg:text-5xl font-bold text-balance text-foreground">
              Our <span className="text-secondary">Services</span>
            </h2>
            <p className="text-lg md:text-xl text-muted-foreground max-w-3xl mx-auto text-pretty leading-relaxed">
              Comprehensive technology solutions designed to accelerate your digital journey and deliver measurable
              business value.
            </p>
          </div>

          <div ref={gridRef} className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 lg:gap-8 scroll-reveal">
            {services.map((service, index) => (
              <Card
                key={index}
                className="p-6 lg:p-8 hover:shadow-xl hover:border-primary/30 transition-all duration-300 group bg-card border-border"
              >
                <div className="space-y-4">
                  {/* Icon */}
                  <div className="h-14 w-14 rounded-lg bg-primary/10 flex items-center justify-center group-hover:bg-primary/20 transition-colors">
                    <service.icon className="h-7 w-7 text-primary" />
                  </div>

                  {/* Content */}
                  <div className="space-y-2">
                    <h3 className="text-xl font-semibold text-foreground group-hover:text-primary transition-colors">
                      {service.title}
                    </h3>
                    <p className="text-muted-foreground leading-relaxed">{service.description}</p>
                  </div>

                  {/* Hover indicator */}
                  <div className="pt-2">
                    <span className="text-sm text-primary font-medium opacity-0 group-hover:opacity-100 transition-opacity inline-flex items-center gap-1">
                      Learn more
                      <svg
                        width="16"
                        height="16"
                        viewBox="0 0 16 16"
                        fill="none"
                        className="group-hover:translate-x-1 transition-transform"
                      >
                        <path
                          d="M6 3L11 8L6 13"
                          stroke="currentColor"
                          strokeWidth="2"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        />
                      </svg>
                    </span>
                  </div>
                </div>
              </Card>
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}
