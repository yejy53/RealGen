"use client";

import { motion } from "framer-motion";
import Image from "next/image";

export default function Comparison() {
    return (
        <section className="py-24 bg-background">
            <div className="container mx-auto px-4">
                <div className="flex items-center gap-4 mb-16">
                    <div className="h-px bg-border flex-1" />
                    <h2 className="text-3xl font-bold tracking-tight uppercase text-foreground">Comparison</h2>
                    <div className="h-px bg-border flex-1" />
                </div>

                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    className="space-y-4 text-center"
                >
                    {/* <p className="text-muted-foreground mb-8">
                        Qualitative comparison of different methods on RealBench.
                    </p> */}

                    <div className="relative rounded-xl overflow-hidden border border-border shadow-2xl bg-secondary/5">
                        <Image
                            src="imgs/compare.jpg"
                            alt="Comparison with other methods"
                            width={1920}
                            height={1080}
                            className="w-full h-auto object-contain"
                            unoptimized
                        />
                    </div>
                </motion.div>
            </div>
        </section>
    );
}
