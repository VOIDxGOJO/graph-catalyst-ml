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
    // put at top if log exists
    const existing = document.querySelector('.log-entries');
    if (existing) existing.prepend(line);
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
    mainResult.innerHTML = `<div class="placeholder"><p class="muted">Run a prediction to see results here.</p></div>`;
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
        setStatus("Prediction complete (agent & solvent shown).");
        log(`Predicted agent: ${pred.agent_smiles ?? pred.agent ?? "N/A"}`);
    } catch (err) {
        console.error(err);
        setStatus("Server unreachable or error. Use demo.", true);
        log(`Error: ${err.message}`);
    }
}

function renderPrediction(pred) {
    // agent and solvent (prefer canonical if present)
    const agent_smiles = pred.agent_smiles || pred.agent || pred.catalyst || "N/A";
    const agent_cano = pred.agent_smiles_canonical || agent_smiles;
    const solvent_smiles = pred.solvent_smiles || pred.solvent || (pred.solvent_prediction && pred.solvent_prediction.solvent) || "N/A";
    const solvent_cano = pred.solvent_smiles_canonical || solvent_smiles;
    const conf = (typeof pred.confidence === 'number') ? Math.round(pred.confidence * 100) + "%" : "N/A";
    const note = pred.protocol_note || "";

    // images base64
    const agent_img = pred.agent_img || null;
    const solvent_img = pred.solvent_img || null;

    // main block for text + images
    mainResult.innerHTML = `
      <div class="prediction">
        <div class="pred-left">
          <div class="pred-agent">${escapeHtml(agent_cano)}</div>
          <div class="pred-confidence">Confidence: ${conf}</div>
          <div class="muted small">${escapeHtml(note)}</div>
          <div style="margin-top:8px;font-size:13px;color:var(--muted)">SMILES: <span id="agent-smiles">${escapeHtml(agent_smiles)}</span>
            <button id="copy-agent" class="btn tiny" style="margin-left:8px">Copy SMILES</button>
          </div>
        </div>

        <div class="pred-right">
          <div class="pred-solvent">${escapeHtml(solvent_cano)}</div>
          <div class="muted small">Predicted solvent</div>
          <div style="margin-top:8px;font-size:13px;color:var(--muted)">T: N/A · time: N/A · yield: N/A</div>
        </div>
      </div>
      <div class="structure-row">
        <div class="structure-card">
          ${agent_img ? `<img id="agent-img" src="data:image/png;base64,${agent_img}" alt="Agent structure" />` : `<div style="width:160px;height:120px;display:flex;align-items:center;justify-content:center;color:var(--muted)">No image</div>`}
          <div style="font-size:13px;color:var(--muted);max-width:220px;word-break:break-word">${escapeHtml(agent_cano)}</div>
        </div>
        <div class="structure-card">
          ${solvent_img ? `<img id="solvent-img" src="data:image/png;base64,${solvent_img}" alt="Solvent structure" />` : `<div style="width:160px;height:120px;display:flex;align-items:center;justify-content:center;color:var(--muted)">No image</div>`}
          <div style="font-size:13px;color:var(--muted);max-width:220px;word-break:break-word">${escapeHtml(solvent_cano)}</div>
        </div>
      </div>
    `;
    resultCard.removeAttribute('aria-hidden');

    // wire copy button
    const copyBtnAgent = document.getElementById('copy-agent');
    if (copyBtnAgent) {
        copyBtnAgent.onclick = () => {
            navigator.clipboard?.writeText(agent_smiles).then(() => log("Copied agent SMILES")).catch(() => log("Copy failed"));
        };
    }
}

function renderAlternatives(list) {
    alternativesEl.innerHTML = "";
    if (!list || !list.length) return;
    alternativesEl.innerHTML = `<h4>Top alternatives</h4>`;
    list.forEach(a => {
        const score = Math.round((a.score || 0.0) * 100);
        const agent = a.agent || a.catalyst || "N/A";
        const solvent = a.solvent || a.solvent_smiles || "-";
        alternativesEl.innerHTML += `
      <div class="alt-item">
        <div style="width:140px; font-family: monospace;">${escapeHtml(agent)}</div>
        <div class="alt-bar"><div class="alt-bar-fill" style="width:${score}%"></div></div>
        <div style="min-width:80px;text-align:right">${escapeHtml(solvent)}</div>
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
        const id = s.id || "RXN";
        const smiles = s.smiles || "";
        const agent = s.agent || s.catalyst || "";
        const solvent = s.solvent || "";
        const yr = (s.yield !== undefined && s.yield !== null) ? `${s.yield}%` : "";
        const t = (s.temperature !== undefined && s.temperature !== null) ? `${s.temperature}°C` : "";
        similarList.innerHTML += `
      <div class="similar-item">
        <div style="max-width:70%"><strong>${escapeHtml(id)}</strong><div class="muted" style="font-size:12px;word-break:break-word">${escapeHtml(smiles)}</div></div>
        <div style="text-align:right"><strong>${escapeHtml(agent)}</strong><div class="muted" style="font-size:12px">${escapeHtml(solvent)}${t ? ' · ' + t : ''}${yr ? ' · yield ' + yr : ''}</div></div>
      </div>`;
    });
}

async function runMockPrediction() {
    setStatus("Demo prediction (no server): conservative outputs only");
    const mock = {
        prediction: {
            agent: "[Pt]",
            agent_smiles: "Clc1ccc(Cl)cc1Cl", // demo SMILES
            agent_smiles_canonical: "Clc1ccc(Cl)cc1Cl",
            agent_img: null,
            solvent: "CCO",
            solvent_img: null,
            confidence: 0.78,
            protocol_note: "Demo"
        },
        alternatives: [
            { agent: "Pd(PPh3)4", score: 0.62, solvent: "THF" },
            { agent: "Pd2(dba)3", score: 0.45, solvent: "DMA" }
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
    const txt = (mainResult.innerText || "").trim();
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

function escapeHtml(unsafe) {
    if (unsafe === null || unsafe === undefined) return "";
    return String(unsafe).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
}

initSamples();
log("Frontend ready: images + canonical SMILES supported.");
