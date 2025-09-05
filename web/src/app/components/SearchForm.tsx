"use client";

import React, { useState } from "react";
import type { PredictPayload } from "../../../lib/api";

interface Props {
    onSubmit: (payload: PredictPayload) => void;
    submitting?: boolean;
}

const SearchForm: React.FC<Props> = ({ onSubmit, submitting = false }) => {
    const [smiles, setSmiles] = useState("");
    const [solvent, setSolvent] = useState("");
    const [temperature, setTemperature] = useState("");

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (!smiles.trim()) return;
        onSubmit({
            reaction_smiles: smiles.trim(),
            solvent: solvent.trim() || undefined,
            temperature: temperature !== "" ? Number(temperature) : undefined,
        });
    };

    return (
        <form onSubmit={handleSubmit} className="grid grid-cols-1 gap-4 sm:grid-cols-3 items-end">
            <div className="sm:col-span-2">
                <label className="block text-sm font-medium text-slate-700 mb-2">Reaction SMILES</label>
                <input
                    value={smiles}
                    onChange={(e) => setSmiles(e.target.value)}
                    placeholder={'e.g. "BrCc1ccc(cc1)B(O)O >> Ph-Br"'}
                    className="w-full rounded-xl border p-3 shadow-sm focus:ring-2 focus:ring-indigo-300 outline-none"
                />
                <p className="text-xs text-slate-400 mt-1">Use reactants &gt;&gt; product (optional) or list reagents separated by " + ".</p>
            </div>

            <div>
                <label className="block text-sm font-medium text-slate-700 mb-2">Solvent (optional)</label>
                <input value={solvent} onChange={(e) => setSolvent(e.target.value)} className="w-full rounded-xl border p-2" placeholder="e.g. THF" />
            </div>

            <div className="sm:col-span-1">
                <label className="block text-sm font-medium text-slate-700 mb-2">Temp (°C)</label>
                <input value={temperature} onChange={(e) => setTemperature(e.target.value)} className="w-full rounded-xl border p-2" placeholder="e.g. 25" />
            </div>

            <div className="sm:col-span-3 flex gap-3 mt-2">
                <button
                    type="submit"
                    disabled={submitting}
                    className="rounded-2xl bg-indigo-600 text-white px-5 py-3 shadow hover:bg-indigo-700 disabled:opacity-60">
                    {submitting ? "Predicting..." : "Predict Catalyst"}
                </button>

                <button
                    type="button"
                    className="rounded-2xl border px-4 py-3 text-sm"
                    onClick={() => {
                        setSmiles("");
                        setSolvent("");
                        setTemperature("");
                    }}>
                    Reset
                </button>
            </div>
        </form>
    );
};

export default SearchForm;
