const API_ENDPOINT = "http://127.0.0.1:8000/predict";

const form = document.getElementById("predict-form");
const smilesInput = document.getElementById("smiles");
const solventInput = document.getElementById("solvent");
const baseInput = document.getElementById("base");
const tempInput = document.getElementById("temp");
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
    { label: "EM 1", smiles: "O=S(=O)(Nc1cccc(-c2cnc3ccccc3n2)c1)c1cccs1", solvent: "", base: "", temp: "25" },
    { label: "EM 2", smiles: "NC(=O)c1ccc2c(c1)nc(C1CCC(O)CC1)n2CCCO", solvent: "", base: "", temp: "25" }
];


function initSamples() {
    SAMPLES.forEach(s => {
        const b = document.createElement('button');
        b.type = "button";
        b.className = "sample-item";
        b.textContent = s.label;
        b.onclick = () => { smilesInput.value = s.smiles; solventInput.value = s.solvent; baseInput.value = s.base; tempInput.value = s.temp; };
        sampleList.appendChild(b);
    });
}

function log(msg) {
    const line = document.createElement('div');
    line.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
    logEntries?.prepend(line);
}

function setStatus(txt, isError = false) {
    statusArea.textContent = txt;
    statusArea.style.color = isError ? "#ffb4b4" : "inherit";
}

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const tempVal = tempInput.value.trim();
    const payload = {
        smiles: smilesInput.value.trim(),
        solvent: solventInput.value.trim(),
        base: baseInput.value.trim(),
        temperature: tempVal === "" ? null : Number(tempVal)
    };
    await runPrediction(payload);
});

clearBtn.addEventListener('click', () => location.reload());
demoBtn.addEventListener('click', () => runMockPrediction(SAMPLES[0]));

async function runPrediction(payload) {
    setStatus("Running prediction...");
    mainResult.innerHTML = `<div class="placeholder"><p class="muted">Waiting for model...</p></div>`;
    try {
        const resp = await fetch(API_ENDPOINT, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
        const data = await resp.json();

        const pred = data.prediction || {};
        const alts = data.alternatives || [];
        const similar = data.similar_reactions || [];

        showPredictionBlock(pred);
        renderAlternatives(alts);
        renderSimilar(similar);
        setStatus("Prediction complete");
        log(`Predicted ${pred.catalyst ?? "N/A"} (${pred.loading_mol_percent ?? "-"} mol%)`);
    } catch (err) {
        setStatus("Server unreachable. Try demo.", true);
        log(`Network error: ${err.message}`);
    }
}

function showPredictionBlock(pred) {
    mainResult.innerHTML = `
    <div class="prediction">
      <div class="pred-left">
        <div class="pred-catalyst">${pred.catalyst ?? "N/A"}</div>
        <div class="pred-confidence">Confidence: ${Math.round((pred.confidence || 0) * 100)}%</div>
        <div class="muted small">${pred.protocol_note || ""}</div>
      </div>
      <div class="pred-right">
        <div class="pred-loading">${(pred.loading_mol_percent ?? 0).toFixed(2)} mol%</div>
        <div class="muted" style="font-size:13px">Suggested</div>
      </div>
    </div>`;
    resultCard.removeAttribute('aria-hidden');
}

function renderAlternatives(list) {
    alternativesEl.innerHTML = "";
    if (!list.length) return;
    alternativesEl.innerHTML = `<h4>Top alternatives</h4>`;
    list.forEach(a => {
        const pct = Math.round((a.score || 0.5) * 100);
        alternativesEl.innerHTML += `<div class="alt-item"><div style="width:140px">${a.catalyst}</div>
      <div class="alt-bar"><div class="alt-bar-fill" style="width:${pct}%"></div></div>
      <div style="min-width:60px;text-align:right">${(a.loading_mol_percent ?? 0).toFixed(2)} mol%</div></div>`;
    });
}

function renderSimilar(list) {
    similarList.innerHTML = "";
    if (!list.length) return;
    list.forEach(s => {
        similarList.innerHTML += `<div class="similar-item">
      <div><strong>${s.id || "RXN"}</strong><div class="muted" style="font-size:12px">${s.smiles}</div></div>
      <div style="text-align:right"><strong>${s.catalyst}</strong><div class="muted" style="font-size:12px">${s.loading} mol%</div></div>
    </div>`;
    });
}

async function runMockPrediction(payload) {
    setStatus("Running demo...");
    const mock = {
        prediction: { catalyst: "Pd(PPh3)4", loading_mol_percent: 2.5, confidence: 0.86, protocol_note: "Dry solvent, inert atmosphere" },
        alternatives: [{ catalyst: "Pd2(dba)3", loading_mol_percent: 3, score: 0.72 }],
        similar_reactions: [{ id: "RXN-001", smiles: "PhBr+PhB(OH)2→Ph-Ph", catalyst: "Pd(PPh3)4", loading: 2.5 }]
    };
    await new Promise(r => setTimeout(r, 600));
    showPredictionBlock(mock.prediction);
    renderAlternatives(mock.alternatives);
    renderSimilar(mock.similar_reactions);
    setStatus("Demo complete");
}

initSamples();
log("Frontend ready");
