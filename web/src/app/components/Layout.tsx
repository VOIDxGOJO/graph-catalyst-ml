"use client";

import React from "react";

const LayoutClient: React.FC<React.PropsWithChildren> = ({ children }) => {
    return (
        <div className="min-h-screen bg-gradient-to-b from-white to-slate-50 text-slate-900">
            <header className="border-b bg-white/60 backdrop-blur-sm">
                <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-indigo-600 flex items-center justify-center text-white font-semibold">CD</div>
                        <div>
                            <div className="text-lg font-semibold">Catalyst Designer</div>
                            <div className="text-xs text-slate-500">Graph-based ML demo</div>
                        </div>
                    </div>
                    <nav className="text-sm text-slate-600">
                        <a className="mr-4 hover:underline" href="#">Home</a>
                        <a className="mr-4 hover:underline" href="#">Docs</a>
                        <a className="hover:underline" href="#">About</a>
                    </nav>
                </div>
            </header>

            <main>{children}</main>

            <footer className="mt-12 py-6 text-center text-xs text-slate-500">
                Built for demo • Not for lab use • Data-driven suggestions only
            </footer>
        </div>
    );
};

export default LayoutClient;
