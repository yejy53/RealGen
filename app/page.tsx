"use client";

import Hero from "@/components/Hero";
import Abstract from "@/components/Abstract";
import Gallery from "@/components/Gallery";
import Comparison from "@/components/Comparison";
import Citation from "@/components/Citation";

export default function Home() {
  return (
    <main className="min-h-screen bg-background text-foreground selection:bg-primary selection:text-primary-foreground">
      <Hero />
      <Abstract />
      <Gallery />
      <Comparison />
      <Citation />
    </main>
  );
}
