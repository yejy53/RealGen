"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X, ZoomIn } from "lucide-react";
import Image from "next/image";

// Sample images assuming they are in public/imgs
// You should verify the actual filenames in public/imgs
const images = Array.from({ length: 40 }, (_, i) => `imgs/图片${i + 1}.png`);

export default function Gallery() {
    const [selectedImage, setSelectedImage] = useState<string | null>(null);

    return (
        <section className="py-24 bg-secondary/5">
            <div className="container mx-auto px-4">
                <div className="flex items-center gap-4 mb-16">
                    <div className="h-px bg-border flex-1" />
                    <h2 className="text-3xl font-bold tracking-tight uppercase text-foreground">Gallery</h2>
                    <div className="h-px bg-border flex-1" />
                </div>

                <div className="columns-2 md:columns-4 lg:columns-5 gap-3 space-y-3">
                    {images.map((src, index) => (
                        <motion.div
                            key={index}
                            initial={{ opacity: 0, y: 20 }}
                            whileInView={{ opacity: 1, y: 0 }}
                            viewport={{ once: true }}
                            transition={{ delay: index * 0.1, duration: 0.5 }}
                            className="relative group break-inside-avoid overflow-hidden rounded-lg cursor-zoom-in border border-border bg-card"
                            onClick={() => setSelectedImage(src)}
                        >
                            <div className="absolute inset-0 bg-black/0 group-hover:bg-black/20 transition-colors z-10 flex items-center justify-center">
                                <ZoomIn className="text-white opacity-0 group-hover:opacity-100 transform scale-50 group-hover:scale-100 transition-all duration-300" />
                            </div>
                            <Image
                                src={src}
                                alt={`Gallery image ${index + 1}`}
                                width={800}
                                height={600}
                                className="w-full h-auto transform group-hover:scale-105 transition-transform duration-500 ease-in-out"
                                unoptimized // Simplified for local images without optimization setup
                            />
                        </motion.div>
                    ))}
                </div>
            </div>

            {/* Lightbox / Zoom View */}
            <AnimatePresence>
                {selectedImage && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="fixed inset-0 z-50 flex items-center justify-center bg-background/95 backdrop-blur-md p-4"
                        onClick={() => setSelectedImage(null)}
                    >
                        <Button
                            variant="ghost"
                            size="icon"
                            className="absolute top-4 right-4 text-foreground/50 hover:text-foreground z-50"
                            onClick={() => setSelectedImage(null)}
                        >
                            <X className="w-8 h-8" />
                        </Button>
                        <motion.div
                            initial={{ scale: 0.9, opacity: 0 }}
                            animate={{ scale: 1, opacity: 1 }}
                            exit={{ scale: 0.9, opacity: 0 }}
                            className="relative w-full max-w-7xl max-h-[90vh] flex items-center justify-center p-2"
                            onClick={(e) => e.stopPropagation()} // Prevent closing when clicking image wrapper
                        >
                            {/* Click on image itself to close as well if desired, or zoom more */}
                            <Image
                                src={selectedImage}
                                alt="Zoomed gallery image"
                                width={1920}
                                height={1080}
                                className="object-contain max-h-[90vh] w-auto h-auto rounded-md shadow-2xl"
                                onClick={() => setSelectedImage(null)} // Click image to close
                                unoptimized
                            />
                        </motion.div>
                    </motion.div>
                )}
            </AnimatePresence>
        </section>
    );
}

// Helper for button inside standard component
import { Button } from "@/components/ui/button";
