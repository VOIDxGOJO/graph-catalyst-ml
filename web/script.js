// fe expects backend POST /predict to return this structure:
// {
//   prediction: {
//     agent: "Pd(PPh3)4",              // top predicted agent (string): primary target
//     solvent: "THF",                 // predicted solvent (string)
//     confidence: 0.82,               // overall confidence (0..1) - optional
//     protocol_note: "retrieved from similar" // optional short note
//     temperature_pred: 25,           // optional numeric (if model returns it)
//     rxn_time_pred: 1.5,             // optional numeric (hours)
//     yield_pred: 78.0                // optional numeric (percent)
//   },
//   alternatives: [
//     { agent: "...", solvent: "...", score: 0.6 }
//   ],
//   similar_reactions: [
//     { id: "461496", smiles: "...", agent: "[Pt]", solvent: "CCO", rxn_time: 0.73, temperature: 67.9, yield: 67.9 }
//   ]
// }
//
// IMP: fe only relies on agent and solvent as guaranteed outputs
// temperature, rxn_time, yield may be empty or missing in many rows (low confidence)

const API_ENDPOINT = "http://127.0.0.1:8000/predict";

const form = document.getElementById("predict-form");
const smilesInput = document.getElementById("smiles");
const productInput = document.getElementById("product");
const tempInput = document.getElementById("temp");
const rxnTimeInput = document.getElementById("rxn_time");
const yieldInput = document.getElementById("yield");
const resultCard = document.getElementById("result-card");
const mainResult = document.getElementById("main-result");
const alternativesEl = document.getElementById("alternatives");
const similarPanel = document.getElementById("similar-panel");
const similarList = document.getElementById("similar-list");
const logEntries = document.getElementById("log-entries");
const statusArea = document.getElementById("status-area");

const demoBtn = document.getElementById("demo-btn");
const sampleList = document.getElementById("sample-list");
const clearBtn = document.getElementById("clear-btn");
const copyBtn = document.getElementById("copy-btn");
const downloadBtn = document.getElementById("download-btn");

const SAMPLES = [
    {
        label: "Hydrogenation (Pt)",
        smiles: "CC(=O)c1cc(C)cc([N+](=O)[O-])c1O>>CC(=O)c1cc(C)cc(N)c1O",
        product: "CC(=O)c1cc(C)cc(N)c1O",
        solvent: "CCO",
        temp: "25",
        rxn_time: "0.73"
    },
    {
        label: "O-alkylation (K t-Bu)",
        smiles: "Nc1ccc(O)cc1F.CC(C)([O-])[K+].Cl[C:17]1...>>COc1cc(Cl)ccn1",
        product: "COc1cc(Cl)ccn1",
        solvent: "DMA",
        temp: "25",
        rxn_time: "1"
    },
    {
        label: "SN2 (NaH alkylation)",
        smiles: "Oc1cccc(C(F)(F)F)n1.CC(C)(C)CBr>>CC(C)(C)COc1cccc(C(F)(F)F)n1",
        product: "CC(C)(C)COc1cccc(C(F)(F)F)n1",
        solvent: "DMF",
        temp: "100",
        rxn_time: "overnight"
    }
];

function initSamples() {
    SAMPLES.forEach(s => {
        const b = document.createElement('button');
        b.type = "button";
        b.className = "sample-item";
        b.textContent = s.label;
        b.onclick = () => {
            smilesInput.value = s.smiles || "";
            productInput.value = s.product || "";
            tempInput.value = s.temp || "";
            rxnTimeInput.value = s.rxn_time || "";
            yieldInput.value = s.yield || "";
        };
        sampleList.appendChild(b);
    });
}

function log(msg) {
    const line = document.createElement('div');
    line.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
    logEntries.prepend(line);
}

function setStatus(txt, isError = false) {
    statusArea.textContent = txt;
    statusArea.style.color = isError ? "#ffb4b4" : "inherit";
}

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const tempVal = tempInput.value.trim();
    const payload = {
        smiles: smilesInput.value.trim() || null,
        product_smiles: productInput.value.trim() || null,
        temperature_c: tempVal === "" ? null : Number(tempVal),
        rxn_time_h: rxnTimeInput.value.trim() === "" ? null : Number(rxnTimeInput.value.trim()),
        yield_percent: yieldInput.value.trim() === "" ? null : Number(yieldInput.value.trim())
    };
    await runPrediction(payload);
});

clearBtn.addEventListener('click', () => {
    smilesInput.value = "";
    productInput.value = "";
    tempInput.value = "";
    rxnTimeInput.value = "";
    yieldInput.value = "";
    mainResult.innerHTML = `<div class="placeholder"><p class="muted">Run a prediction to see agent & solvent here.</p></div>`;
    alternativesEl.innerHTML = "";
    similarList.innerHTML = "";
    resultCard.setAttribute('aria-hidden', 'true');
    setStatus("");
});

demoBtn.addEventListener('click', () => runMockPrediction());

async function runPrediction(payload) {
    setStatus("Running prediction...");
    mainResult.innerHTML = `<div class="placeholder"><p class="muted">Waiting for model...</p></div>`;
    resultCard.setAttribute('aria-hidden', 'true');

    try {
        const resp = await fetch(API_ENDPOINT, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (!resp.ok) {
            const text = await resp.text();
            throw new Error(`Server error ${resp.status}: ${text}`);
        }

        const data = await resp.json();
        const pred = data.prediction || {};
        const alts = data.alternatives || [];
        const similar = data.similar_reactions || [];

        renderPrediction(pred);
        renderAlternatives(alts);
        renderSimilar(similar);
        setStatus("Prediction complete (agent & solvent are the primary outputs)");
        log(`Predicted agent: ${pred.agent ?? "N/A"}, solvent: ${pred.solvent ?? "N/A"}`);
    } catch (err) {
        console.error(err);
        setStatus("Server unreachable or error. Use demo.", true);
        log(`Error: ${err.message}`);
    }
}

function renderPrediction(pred) {
    // use agent and solvent as guaranteed fields
    const agent = escapeHtml(pred.agent ?? "N/A");
    const solvent = escapeHtml(pred.solvent ?? "N/A");
    const conf = (typeof pred.confidence === 'number') ? Math.round(pred.confidence * 100) + "%" : "N/A";
    const note = escapeHtml(pred.protocol_note || "");
    const tempPred = (pred.temperature_pred !== undefined && pred.temperature_pred !== null) ? `${pred.temperature_pred} °C` : "N/A";
    const timePred = (pred.rxn_time_pred !== undefined && pred.rxn_time_pred !== null) ? `${pred.rxn_time_pred} h` : "N/A";
    const yieldPred = (pred.yield_pred !== undefined && pred.yield_pred !== null) ? `${pred.yield_pred}%` : "N/A";

    mainResult.innerHTML = `
    <div class="prediction">
      <div class="pred-left">
        <div class="pred-agent">${agent}</div>
        <div class="pred-confidence">Confidence: ${conf}</div>
        <div class="muted small">${note}</div>
      </div>
      <div class="pred-right">
        <div class="pred-solvent">${solvent}</div>
        <div class="muted small">Predicted solvent</div>
        <div style="margin-top:8px;font-size:13px;color:var(--muted)">T: ${tempPred} · time: ${timePred} · yield: ${yieldPred}</div>
      </div>
    </div>
  `;
    resultCard.removeAttribute('aria-hidden');
}

function renderAlternatives(list) {
    alternativesEl.innerHTML = "";
    if (!list || !list.length) return;
    alternativesEl.innerHTML = `<h4>Top alternatives</h4>`;
    list.forEach(a => {
        const score = Math.round((a.score || 0.5) * 100);
        const agent = escapeHtml(a.agent || 'N/A');
        const solvent = escapeHtml(a.solvent || '-');
        alternativesEl.innerHTML += `
      <div class="alt-item">
        <div style="width:140px">${agent}</div>
        <div class="alt-bar"><div class="alt-bar-fill" style="width:${score}%"></div></div>
        <div style="min-width:80px;text-align:right">${solvent}</div>
      </div>`;
    });
}

function renderSimilar(list) {
    similarList.innerHTML = "";
    if (!list || !list.length) {
        similarPanel.setAttribute('aria-hidden', 'true');
        return;
    }
    similarPanel.removeAttribute('aria-hidden');
    list.forEach(s => {
        const id = escapeHtml(s.id || "RXN");
        const smiles = escapeHtml(s.smiles || "");
        const agent = escapeHtml(s.agent || "");
        const solvent = escapeHtml(s.solvent || "");
        const yr = (s.yield !== undefined && s.yield !== null) ? `${s.yield}%` : "";
        const t = (s.temperature !== undefined && s.temperature !== null) ? `${s.temperature}°C` : "";
        similarList.innerHTML += `
      <div class="similar-item">
        <div style="max-width:70%"><strong>${id}</strong><div class="muted" style="font-size:12px;word-break:break-word">${smiles}</div></div>
        <div style="text-align:right"><strong>${agent}</strong><div class="muted" style="font-size:12px">${solvent}${t ? ' · ' + t : ''}${yr ? ' · yield ' + yr : ''}</div></div>
      </div>`;
    });
}

async function runMockPrediction() {
    setStatus("Demo prediction (no server): conservative outputs only");
    const mock = {
        prediction: {
            agent: "[Pt] (Pt on charcoal)",
            solvent: "CCO",
            confidence: 0.78,
            protocol_note: "Retrieved from similar literature example",
            temperature_pred: 25,
            rxn_time_pred: 0.73,
            yield_pred: 67.9
        },
        alternatives: [
            { agent: "Pd(PPh3)4", solvent: "THF", score: 0.62 },
            { agent: "Pd2(dba)3", solvent: "DMA", score: 0.45 }
        ],
        similar_reactions: [
            { id: "439257", smiles: "CC(=O)c1cc...>>...", agent: "[Pt]", solvent: "CCO", temperature: 25, rxn_time: 0.73, yield: 67.9 }
        ]
    };

    await new Promise(r => setTimeout(r, 400));
    renderPrediction(mock.prediction);
    renderAlternatives(mock.alternatives);
    renderSimilar(mock.similar_reactions);
    setStatus("Demo complete (agent & solvent shown).");
    log("Demo: predicted agent & solvent (conservative).");
}

copyBtn.addEventListener('click', () => {
    const txt = mainResult.innerText || "";
    if (!navigator.clipboard) { log("Clipboard not available"); return; }
    navigator.clipboard.writeText(txt).then(() => log("Copied prediction to clipboard")).catch(() => log("Copy failed"));
});

downloadBtn.addEventListener('click', () => {
    const text = mainResult.innerText || "No prediction";
    const data = { text, timestamp: new Date().toISOString() };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'prediction.json';
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(a.href);
    log("Downloaded prediction.json");
});

// escape to avoid putting arbitrary text
function escapeHtml(unsafe) {
    if (unsafe === null || unsafe === undefined) return "";
    return String(unsafe).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
}

initSamples();
log("Frontend ready: conservative (agent + solvent only).");
