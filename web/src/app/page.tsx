"use client";

import React, { useState } from "react";
import LayoutClient from "./components/Layout";
import SearchForm from "./components/SearchForm";
import ResultCard from "./components/ResultCard";
import type { PredictPayload, PredictResponse } from "../../lib/api";
import { predictReaction } from "../../lib/api";

export default function Page() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const onSubmit = async (payload: PredictPayload) => {
    setError(null);
    setResult(null);
    setLoading(true);
    try {
      const res = await predictReaction(payload);
      setResult(res);
    } catch (err: any) {
      console.error(err);
      setError(err?.message || "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  return (
    <LayoutClient>
      <div className="max-w-4xl mx-auto py-12 px-4 sm:px-6 lg:px-8">
        <h1 className="text-3xl font-semibold mb-2">Catalyst Designer</h1>
        <p className="text-sm text-slate-500 mb-6">
          Paste reaction SMILES or provide reactants; this model will suggest a catalyst and a recommended loading (mol %).
        </p>

        <SearchForm onSubmit={onSubmit} submitting={loading} />

        {error && (
          <div className="mt-6 p-4 rounded-lg bg-red-50 border border-red-200 text-red-700">{error}</div>
        )}

        {result && (
          <div className="mt-6">
            <ResultCard result={result} />
          </div>
        )}

        {!result && !error && (
          <div className="mt-8 text-sm text-slate-400">
            Tip: try a Suzuki coupling example like <code className="px-1 rounded bg-slate-100">BrCc1ccc(cc1)B(O)O</code>
          </div>
        )}
      </div>
    </LayoutClient>
  );
}
