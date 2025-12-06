"use client";

import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Github, FileText, ArrowUpRight } from "lucide-react";

const authors = [
    { name: "Junyan Ye", affiliations: [1, 2], isEqual: true, url: "https://yejy53.github.io/" },
    { name: "Leiqi Zhu", affiliations: [1, 3], isEqual: true, url: "#" },
    { name: "Yuncheng Guo", affiliations: [1], url: "#" },
    { name: "Dongzhi Jiang", affiliations: [4], url: "https://caraj7.github.io/" },
    { name: "Zilong Huang", affiliations: [2], url: "https://serein-six.vercel.app/" },
    { name: "Yifan Zhang", affiliations: [5], url: "#" },
    { name: "Zhiyuan Yan", affiliations: [6], url: "https://yzy-stack.github.io/" },
    { name: "Haohuan Fu", affiliations: [5], url: "https://www.sigs.tsinghua.edu.cn/fhh_en/main.htm" },
    { name: "Conghui He", affiliations: [1], url: "https://conghui.github.io/" },
    { name: "Weijia Li", affiliations: [2], isCorresponding: true, url: "https://liweijia.github.io/" },
];

const affiliations = [
    "Shanghai AI Lab",
    "Sun Yat-Sen University",
    "Nanjing University",
    "CUHK MMLab",
    "Tsinghua University",
    "Peking University",
];

export default function Hero() {
    return (
        <section className="relative min-h-[50vh] w-full overflow-hidden flex flex-col items-center justify-center text-center bg-white selection:bg-red-500/30 py-12">
            {/* 1. Aurora Background Effect - STRICT WHITE/BLUE */}
            <div className="absolute inset-0 pointer-events-none overflow-hidden">
                <div
                    className="absolute -inset-[10px] opacity-20 blur-[60px]"
                    style={{
                        background: 'radial-gradient(circle at 50% 50%, rgba(255,255,255,0) 0%, rgba(200,200,200,0.1) 100%)'
                    }}
                />

                {/* Pure White & Very Light Blue Gradients - No other colors */}
                <div
                    className="absolute inset-0 opacity-60 will-change-transform animate-aurora"
                    style={{
                        backgroundImage: `
                            repeating-linear-gradient(100deg, #ffffff 10%, #f0f9ff 20%, #eff6ff 30%, #ffffff 40%),
                            repeating-linear-gradient(100deg, #f0f9ff 10%, #eff6ff 20%, #dbeafe 30%, #f0f9ff 40%)
                        `,
                        backgroundSize: '200% 200%'
                    }}
                />
            </div>

            {/* 2. Base Grid Layer (Subtle Grey on White) */}
            <div className="absolute inset-0 bg-[linear-gradient(to_right,#8080800a_1px,transparent_1px),linear-gradient(to_bottom,#8080800a_1px,transparent_1px)] bg-[size:40px_40px] pointer-events-none" />

            {/* 3. Radial Vignette (White to Transparent) to merge content */}
            <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,transparent_0%,white_95%)] pointer-events-none" />

            {/* 4. Scrolling Text - RealGen Only */}
            <div className="absolute inset-0 opacity-[0.03] pointer-events-none select-none overflow-hidden">
                <div className="absolute inset-0 flex flex-col gap-12 -rotate-12 scale-150">
                    {Array.from({ length: 20 }).map((_, i) => (
                        <div
                            key={i}
                            className="whitespace-nowrap text-9xl font-black text-black animate-scroll-text tracking-widest uppercase"
                            style={{
                                animationDuration: `${30 + i % 5 * 3}s`,
                                animationDirection: i % 2 === 0 ? "normal" : "reverse",
                            }}
                        >
                            RealGen RealGen RealGen RealGen RealGen
                        </div>
                    ))}
                </div>
            </div>

            {/* Content */}
            <div className="relative z-10 container mx-auto px-4 space-y-6">
                <motion.div
                    initial={{ opacity: 0, y: 30 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 1, ease: "easeOut" }}
                    className="space-y-2"
                >
                    {/* <div className="inline-block px-4 py-1.5 mb-6 text-sm font-bold tracking-wider font-mono border border-red-500/20 text-red-600 rounded-full bg-red-50/50 backdrop-blur-sm shadow-sm ring-1 ring-red-100">
                        ArXiv 2025
                    </div> */}
                    <h1 className="text-6xl md:text-8xl lg:text-9xl font-black tracking-tighter mb-4 bg-clip-text text-transparent bg-gradient-to-b from-foreground via-foreground to-foreground/50">
                        RealGen
                    </h1>
                    <p className="text-2xl md:text-3xl text-muted-foreground font-light max-w-4xl mx-auto leading-relaxed">
                        Photorealistic Text-to-Image Generation via Detector-Guided Rewards
                    </p>
                </motion.div>

                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.4, duration: 1 }}
                    className="space-y-6"
                >
                    {/* Authors - Constrained width to force roughly 2 lines */}
                    <div className="flex flex-wrap justify-center gap-x-8 gap-y-2 text-lg md:text-xl font-light max-w-[850px] mx-auto leading-relaxed">
                        {authors.map((author, index) => (
                            <a
                                key={index}
                                href={author.url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="group relative cursor-pointer hover:text-primary transition-colors flex items-center gap-0.5"
                            >
                                {author.name}
                                <sup className="text-xs text-muted-foreground group-hover:text-primary transition-colors">
                                    {author.affiliations.join(",")}
                                    {author.isEqual && "*"}
                                    {author.isCorresponding && "†"}
                                </sup>
                            </a>
                        ))}
                    </div>

                    {/* Affiliations */}
                    <div className="flex flex-wrap justify-center gap-x-6 gap-y-1 text-sm md:text-base text-muted-foreground/80 font-mono max-w-4xl mx-auto">
                        {affiliations.map((aff, index) => (
                            <div key={index} className="flex items-center gap-1">
                                <span className="text-primary/60">{index + 1}</span>
                                {aff}
                            </div>
                        ))}
                    </div>

                    <div className="text-sm text-muted-foreground mt-2 font-mono">
                        * Equal Contribution, † Corresponding Author
                    </div>
                </motion.div>

                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.6, duration: 0.8 }}
                    className="flex flex-col sm:flex-row justify-center gap-4 pt-4"
                >
                    <Button size="lg" className="h-12 px-8 text-lg gap-2 rounded-full font-medium shadow-lg shadow-primary/20 hover:shadow-primary/40 transition-all" asChild>
                        <a href="https://arxiv.org/abs/2512.00473" target="_blank" rel="noopener noreferrer">
                            <FileText className="w-5 h-5" />
                            Paper
                            <ArrowUpRight className="w-4 h-4 opacity-50" />
                        </a>
                    </Button>
                    <Button size="lg" variant="outline" className="h-12 px-8 text-lg gap-2 rounded-full border-primary/20 hover:bg-primary/5 hover:border-primary/50 transition-all" asChild>
                        <a href="https://github.com/yejy53/RealGen?tab=readme-ov-file" target="_blank" rel="noopener noreferrer">
                            <Github className="w-5 h-5" />
                            Code
                            <ArrowUpRight className="w-4 h-4 opacity-50" />
                        </a>
                    </Button>
                </motion.div>
            </div>

            {/* Scroll Indicator */}
            {/* <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 1.5, duration: 1 }}
                className="absolute bottom-8 left-1/2 -translate-x-1/2 text-muted-foreground animate-bounce"
            >
                <div className="w-px h-12 bg-gradient-to-b from-transparent via-muted-foreground to-transparent" />
            </motion.div> */}
        </section>
    );
}

