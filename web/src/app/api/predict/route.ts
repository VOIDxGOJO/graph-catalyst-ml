import { NextResponse } from "next/server";

export async function POST(req: Request) {
    try {
        const payload = await req.json().catch(() => ({}));
        const smiles = (payload?.reaction_smiles || "").toLowerCase();

        // small heuristic to simulate different outputs
        const isSuzuki = smiles.includes("b(") || smiles.includes("bor") || smiles.includes("b(o)");
        const isHeck = smiles.includes("c(=c)") || smiles.includes("alkene");

        const result = isSuzuki
            ? {
                predicted_catalyst: "Pd(PPh3)4",
                loading_mol_percent: 2.5,
                top_alternatives: [
                    { catalyst: "Pd2(dba)3", score: 0.82 },
                    { catalyst: "Pd(OAc)2 / PPh3", score: 0.66 },
                ],
                nearest_reactions: [{ id: "r1", smiles: "Ph-Br + B(OH)2", catalyst: "Pd(PPh3)4", loading: 2.5 }],
            }
            : isHeck
                ? {
                    predicted_catalyst: "Pd(OAc)2",
                    loading_mol_percent: 3.0,
                    top_alternatives: [
                        { catalyst: "PdCl2(PPh3)2", score: 0.58 },
                        { catalyst: "Pd(PPh3)2Cl2", score: 0.43 },
                    ],
                    nearest_reactions: [{ id: "r3", smiles: "Ar-X + alkene", catalyst: "Pd(OAc)2", loading: 3.0 }],
                }
                : {
                    predicted_catalyst: "Pd(OAc)2",
                    loading_mol_percent: 5,
                    top_alternatives: [
                        { catalyst: "NiCl2(dppp)", score: 0.45 },
                        { catalyst: "CuI", score: 0.33 },
                    ],
                    nearest_reactions: [{ id: "r2", smiles: "R-Br + R'-H", catalyst: "Pd(OAc)2", loading: 5 }],
                };

        return NextResponse.json(result);
    } catch (err) {
        return NextResponse.json({ error: "Invalid request" }, { status: 400 });
    }
}
