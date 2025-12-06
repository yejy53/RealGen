"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Copy, Check } from "lucide-react";

const bibtex = `@article{ye2025realgen,
  title={RealGen: Photorealistic Text-to-Image Generation via Detector-Guided Rewards},
  author={Ye, Junyan and Zhu, Leiqi and Guo, Yuncheng and Jiang, Dongzhi and Huang, Zilong and Zhang, Yifan and Yan, Zhiyuan and Fu, Haohuan and He, Conghui and Li, Weijia},
  journal={arXiv preprint arXiv:2512.00473},
  year={2025}
}`;

export default function Citation() {
    const [copied, setCopied] = useState(false);

    const handleCopy = () => {
        navigator.clipboard.writeText(bibtex);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    return (
        <section className="py-24 bg-secondary/5 border-t border-border">
            <div className="container mx-auto px-4 max-w-4xl">
                <div className="flex items-center gap-4 mb-8">
                    <h2 className="text-2xl font-bold tracking-tight uppercase text-foreground">Citation</h2>
                    <div className="h-px bg-border flex-1" />
                </div>

                <motion.div
                    initial={{ opacity: 0 }}
                    whileInView={{ opacity: 1 }}
                    viewport={{ once: true }}
                    className="relative bg-card border border-border rounded-lg p-6 shadow-sm overflow-hidden"
                >
                    <pre className="font-mono text-sm md:text-base whitespace-pre-wrap text-muted-foreground overflow-x-auto">
                        {bibtex}
                    </pre>

                    <Button
                        size="icon"
                        variant="ghost"
                        className="absolute top-2 right-2 hover:bg-muted"
                        onClick={handleCopy}
                    >
                        {copied ? <Check className="w-4 h-4 text-green-500" /> : <Copy className="w-4 h-4 text-muted-foreground" />}
                    </Button>
                </motion.div>
            </div>
        </section>
    );
}
