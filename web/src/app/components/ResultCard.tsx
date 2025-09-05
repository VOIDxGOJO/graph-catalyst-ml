"use client";

import React from "react";
import type { PredictResponse } from "../../../lib/api";

interface Props {
    result: PredictResponse;
}

const ResultCard: React.FC<Props> = ({ result }) => {
    return (
        <div className="rounded-2xl border p-6 bg-white shadow-sm">
            <div className="flex items-start justify-between">
                <div>
                    <div className="text-sm text-slate-500">Recommended catalyst</div>
                    <div className="text-2xl font-semibold mt-1">{result.predicted_catalyst}</div>
                    <div className="mt-2 text-sm text-slate-600">
                        Suggested loading: <span className="font-medium">{result.loading_mol_percent}%</span>
                    </div>
                </div>
                <div className="text-right">
                    <div className="text-xs text-slate-400">Confidence</div>
                    <div className="mt-2 text-lg font-semibold">—</div>
                </div>
            </div>

            {result.top_alternatives && result.top_alternatives.length > 0 && (
                <div className="mt-6">
                    <div className="text-sm text-slate-500 mb-2">Top alternatives</div>
                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                        {result.top_alternatives.map((a, i) => (
                            <div key={i} className="p-3 rounded-xl border">
                                <div className="text-sm font-medium">{a.catalyst}</div>
                                <div className="text-xs text-slate-400">score: {a.score?.toFixed(3) ?? "—"}</div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {result.nearest_reactions && result.nearest_reactions.length > 0 && (
                <div className="mt-6">
                    <div className="text-sm text-slate-500 mb-2">Similar reactions from dataset</div>
                    <div className="space-y-2">
                        {result.nearest_reactions.map((r, idx) => (
                            <div key={idx} className="rounded-xl p-3 border bg-slate-50">
                                <div className="text-xs text-slate-600">{r.smiles}</div>
                                <div className="text-sm font-medium">
                                    catalyst: {r.catalyst} {r.loading ? `• ${r.loading}%` : ""}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
};

export default ResultCard;
