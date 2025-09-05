export interface PredictPayload {
    reaction_smiles: string;
    solvent?: string;
    temperature?: number;
}

export interface PredictResponse {
    predicted_catalyst: string;
    loading_mol_percent: number;
    top_alternatives?: Array<{ catalyst: string; score?: number }>;
    nearest_reactions?: Array<{ id?: string; smiles: string; catalyst: string; loading?: number }>;
}

/**
 * predictReaction
 * - Calls local Next API route /api/predict (mock included above).
 * - Throws on non-2xx or malformed response.
 */
export async function predictReaction(payload: PredictPayload): Promise<PredictResponse> {
    const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    });

    if (!res.ok) {
        const text = await res.text().catch(() => "");
        throw new Error(text || `API error ${res.status}`);
    }

    const data = await res.json().catch(() => null);
    if (!data || typeof data.predicted_catalyst !== "string" || typeof data.loading_mol_percent !== "number") {
        throw new Error("Malformed response from API");
    }

    return data as PredictResponse;
}
