/* ────────────────────────────────────────────────────────────────────
   Digital Health Twin – Frontend Logic
   ────────────────────────────────────────────────────────────────────
   Works in TWO modes:
     1. LIVE MODE  → API running at localhost:8000 (local dev)
     2. DEMO MODE  → client-side prediction engine (Vercel deploy)
   The mode is auto-detected on page load.
   ──────────────────────────────────────────────────────────────────── */

let API = "";
let DEMO_MODE = false;

// Auto-detect: try the local API first, fallback to demo mode
(async function detectMode() {
    const statusPill = document.getElementById("apiStatusPill");
    const statusText = document.getElementById("apiStatusText");
    try {
        const r = await fetch("http://localhost:8000/health", { signal: AbortSignal.timeout(2000) });
        if (r.ok) {
            API = "http://localhost:8000";
            DEMO_MODE = false;
            if (statusText) { statusText.textContent = "Live API Online"; statusPill.classList.add("online"); }
            console.log("Mode: LIVE (connected to FastAPI)");
            return;
        }
    } catch (_) { /* API not available */ }
    DEMO_MODE = true;
    if (statusText) { statusText.textContent = "Demo Mode (Offline)"; statusPill.classList.add("demo"); }
    console.log("Mode: DEMO (client-side engine)");
})();

/* ── Profiles & UI constants ─────────────────────────────────────── */

const BUTTON_LABELS = {
    predict: `${icon("activity", "btn-icon")}<span>Predict Risk</span>`,
    predicting: `${icon("activity", "btn-icon")}<span>Running Prediction...</span>`,
    insight: `${icon("spark", "btn-icon")}<span>Generate Full Insight</span>`,
    generating: `${icon("spark", "btn-icon")}<span>Generating Insight...</span>`,
};

const PROFILES = {
    healthy:  { age: 38, sex: 0, cp: 0, trestbps: 115, chol: 190, fbs: 0, restecg: 0, thalach: 172, exang: 0, oldpeak: 0.2, slope: 0, ca: 0, thal: 1, stress_level: 3, sleep_quality: 8.5, activity_hours: 7, anxiety_score: 3, bmi: 22.5 },
    moderate: { age: 52, sex: 1, cp: 1, trestbps: 132, chol: 235, fbs: 0, restecg: 1, thalach: 148, exang: 0, oldpeak: 1.2, slope: 1, ca: 1, thal: 2, stress_level: 6, sleep_quality: 5.5, activity_hours: 3, anxiety_score: 9, bmi: 27.5 },
    high:     { age: 62, sex: 1, cp: 3, trestbps: 155, chol: 290, fbs: 1, restecg: 2, thalach: 115, exang: 1, oldpeak: 3.1, slope: 2, ca: 3, thal: 3, stress_level: 9, sleep_quality: 3, activity_hours: 1, anxiety_score: 17, bmi: 33.5 },
    mental:   { age: 45, sex: 0, cp: 2, trestbps: 128, chol: 215, fbs: 0, restecg: 0, thalach: 155, exang: 0, oldpeak: 0.8, slope: 1, ca: 0, thal: 2, stress_level: 9.5, sleep_quality: 2.5, activity_hours: 1.5, anxiety_score: 19, bmi: 26 },
};

let currentProfile = "healthy";

/* ── Tab Navigation ──────────────────────────────────────────────── */

document.querySelectorAll(".nav-link").forEach(link => {
    link.addEventListener("click", e => {
        e.preventDefault();
        const tab = link.dataset.tab;
        document.querySelectorAll(".nav-link").forEach(item => item.classList.remove("active"));
        document.querySelectorAll(".tab-content").forEach(item => item.classList.remove("active"));
        link.classList.add("active");
        document.getElementById(tab).classList.add("active");
    });
});

/* ── Utility Functions ───────────────────────────────────────────── */

function icon(name, className = "") {
    const classes = ["icon", className].filter(Boolean).join(" ");
    return `<svg class="${classes}" aria-hidden="true"><use href="#icon-${name}"></use></svg>`;
}

function loadProfile(name) {
    currentProfile = name;
    document.querySelectorAll(".profile-btn").forEach(button => button.classList.remove("active"));
    document.getElementById(`profile-${name}`).classList.add("active");
}

function getFormData() {
    return {
        age: +document.getElementById("age").value,
        sex: +document.getElementById("sex").value,
        cp: +document.getElementById("cp").value,
        trestbps: +document.getElementById("trestbps").value,
        chol: +document.getElementById("chol").value,
        fbs: +document.getElementById("fbs").value,
        restecg: +document.getElementById("restecg").value,
        thalach: +document.getElementById("thalach").value,
        exang: +document.getElementById("exang").value,
        oldpeak: +document.getElementById("oldpeak").value,
        slope: +document.getElementById("slope").value,
        ca: +document.getElementById("ca").value,
        thal: +document.getElementById("thal").value,
        stress_level: +document.getElementById("stress_level").value,
        sleep_quality: +document.getElementById("sleep_quality").value,
        activity_hours: +document.getElementById("activity_hours").value,
        anxiety_score: +document.getElementById("anxiety_score").value,
        bmi: +document.getElementById("bmi").value,
    };
}

function loadingHTML() {
    return `<div class="loading">${icon("activity")}<span>Analyzing records</span><span class="loading-dots"><span></span><span></span><span></span></span></div>`;
}

function getRiskColor(probability) {
    if (probability > 0.6) return "var(--red)";
    if (probability > 0.4) return "var(--amber)";
    return "var(--green)";
}

function getRiskClass(risk) {
    const n = String(risk).toLowerCase();
    if (n.includes("high")) return "risk-high";
    if (n.includes("medium") || n.includes("moderate")) return "risk-medium";
    return "risk-low";
}

function capitalize(v) { return v.charAt(0).toUpperCase() + v.slice(1); }

function escapeHtml(v) {
    const d = document.createElement("div");
    d.textContent = v;
    return d.innerHTML;
}

/* ═══════════════════════════════════════════════════════════════════
   CLIENT-SIDE DEMO ENGINE
   ═══════════════════════════════════════════════════════════════════
   A lightweight JS implementation of the prediction + insight logic
   so the dashboard is fully functional on Vercel without a backend.
   ═══════════════════════════════════════════════════════════════════ */

function demoPredict(data) {
    let score = 0;
    // Weighted risk scoring based on clinical thresholds
    if (data.chol > 240) score += 15; else if (data.chol > 200) score += 7;
    if (data.trestbps > 140) score += 12; else if (data.trestbps > 120) score += 5;
    if (data.bmi > 30) score += 10; else if (data.bmi > 25) score += 4;
    if (data.stress_level > 7) score += 10; else if (data.stress_level > 5) score += 4;
    if (data.anxiety_score > 14) score += 10; else if (data.anxiety_score > 9) score += 5;
    if (data.sleep_quality < 4) score += 8; else if (data.sleep_quality < 6) score += 3;
    if (data.activity_hours < 2) score += 6; else if (data.activity_hours < 4) score += 2;
    if (data.age > 60) score += 8; else if (data.age > 50) score += 4;
    if (data.ca > 0) score += data.ca * 8;
    if (data.thal === 3) score += 10;
    if (data.exang === 1) score += 8;
    if (data.oldpeak > 2) score += 8; else if (data.oldpeak > 1) score += 4;
    if (data.cp === 3) score += 5;
    if (data.thalach < 120) score += 5;

    const probability = Math.min(Math.max(score / 100, 0.01), 0.99);
    const prediction = probability > 0.5 ? 1 : 0;
    const riskLabel = prediction === 1 ? "High Risk" : "Low Risk";
    const riskLevel = probability > 0.8 ? "critical" : probability > 0.6 ? "high" : probability > 0.4 ? "moderate" : "low";

    // Identify factors
    const factors = [];
    if (data.chol > 240) factors.push({ factor: "High cholesterol", value: `${data.chol} mg/dL`, severity: "high", detail: "Cholesterol above 240 mg/dL significantly increases cardiovascular risk." });
    if (data.trestbps > 140) factors.push({ factor: "Hypertension", value: `${data.trestbps} mmHg`, severity: "high", detail: "Resting BP above 140 mmHg indicates hypertension." });
    if (data.bmi > 30) factors.push({ factor: "Obesity", value: `BMI ${data.bmi}`, severity: "high", detail: "BMI above 30 classified as obese." });
    if (data.stress_level > 7) factors.push({ factor: "High stress", value: `${data.stress_level}/10`, severity: "high", detail: "Elevated stress increases cardiovascular risk." });
    if (data.anxiety_score > 14) factors.push({ factor: "Severe anxiety", value: `GAD-7: ${data.anxiety_score}/21`, severity: "high", detail: "Severe anxiety strongly correlates with cardiac events." });
    if (data.sleep_quality < 4) factors.push({ factor: "Poor sleep quality", value: `${data.sleep_quality}/10`, severity: "high", detail: "Poor sleep strongly linked to cardiovascular disease." });
    if (data.activity_hours < 2) factors.push({ factor: "Sedentary lifestyle", value: `${data.activity_hours} hrs/week`, severity: "moderate", detail: "Less than 2 hours/week of activity increases risk." });
    if (data.age > 60) factors.push({ factor: "Advanced age", value: `${data.age} years`, severity: "moderate", detail: "Age above 60 is an independent risk factor." });
    if (data.chol > 200 && data.chol <= 240) factors.push({ factor: "Borderline cholesterol", value: `${data.chol} mg/dL`, severity: "moderate", detail: "Cholesterol 200-240 mg/dL warrants monitoring." });

    return {
        prediction, risk_label: riskLabel, risk_level: riskLevel,
        probability: +probability.toFixed(4),
        confidence: +Math.max(probability, 1 - probability).toFixed(4),
        model_used: "ClientSideEngine (Demo)",
        contributing_factors: factors,
    };
}

function demoAsk(query) {
    const q = query.toLowerCase();
    let response, intent;

    if (/risk|danger|disease|predict|chance/.test(q)) {
        intent = "risk";
        response = "Based on analysis of 5 similar patient profiles, 3/5 (60%) have diagnosed heart disease. Key risk factors include: elevated BMI (30.5), high stress levels (6.8/10), significant anxiety (12/21). Average age in this cohort: 57 years. Given the high prevalence among similar patients, comprehensive cardiovascular screening is strongly recommended.";
    } else if (/mental|stress|anxi|sleep|depress/.test(q)) {
        intent = "mental";
        response = "Mental health analysis across 5 retrieved patient profiles: Average stress level: 7.1/10. Average anxiety score (GAD-7): 12/21. Research demonstrates a strong bidirectional relationship between mental health and cardiovascular risk -- elevated stress and anxiety can increase cardiac event probability by 1.5-2x through inflammation and autonomic dysfunction. Integrated mental-physical health monitoring is recommended.";
    } else if (/treat|recommend|prevent|help|improv|should/.test(q)) {
        intent = "recommendation";
        response = "Based on patterns from 5 similar patient profiles (60% diagnosed with heart disease), the following is recommended: maintain a healthy weight through balanced diet and regular exercise; implement stress management techniques such as meditation, therapy, or structured relaxation; aim for 150+ minutes per week of moderate physical activity; monitor blood pressure and cholesterol levels regularly. Early intervention and lifestyle modifications can significantly reduce cardiovascular risk.";
    } else if (/cause|why|factor|reason/.test(q)) {
        intent = "cause";
        response = "Analysis of 5 similar patient records reveals several contributing factors to cardiovascular risk: elevated BMI (30.5), high stress levels (6.8/10), significant anxiety (12/21). Heart disease is multifactorial -- traditional risk factors (cholesterol, blood pressure, age) interact with lifestyle factors (physical activity, BMI) and mental health indicators (stress, anxiety, sleep quality). The combination of physical and psychological stressors creates compounding risk through chronic inflammation.";
    } else if (/how many|statistic|percent|average|prevalence/.test(q)) {
        intent = "statistics";
        response = "Statistics from 5 retrieved patient profiles: Heart disease prevalence: 60% (3/5). Age range: 48-67 years (avg 57). Average BMI: 30.5. Average stress level: 6.8/10. Average anxiety score: 12/21. These statistics are derived from the most semantically similar patient records in the database to your query.";
    } else {
        intent = "general";
        response = "Based on 5 relevant patient records retrieved from the knowledge base: 3/5 (60%) have diagnosed heart disease. Key observations: elevated BMI (30.5), high stress levels (6.8/10). Average age: 57 years, average BMI: 30.5. Both physical biomarkers and mental health indicators should be considered for comprehensive health assessment.";
    }

    const sampleSources = [
        { patient_id: "P0155", similarity: 0.5139, age: 70, risk: "High" },
        { patient_id: "P0285", similarity: 0.5113, age: 58, risk: "High" },
        { patient_id: "P0100", similarity: 0.5109, age: 54, risk: "Low" },
        { patient_id: "P0271", similarity: 0.5106, age: 66, risk: "Low" },
        { patient_id: "P0073", similarity: 0.5104, age: 65, risk: "High" },
    ];

    return { query, response, sources: sampleSources, num_sources: 5 };
}

function demoInsight(data) {
    const pred = demoPredict(data);
    const factors = pred.contributing_factors;

    // Build narrative
    let causePhrase = factors.slice(0, 3).map(f => {
        if (f.factor === "High cholesterol") return `elevated cholesterol (${f.value})`;
        if (f.factor === "Hypertension") return `hypertension (BP ${f.value})`;
        if (f.factor === "Obesity") return `elevated BMI (${f.value})`;
        if (f.factor === "High stress") return `high stress indicators (${f.value})`;
        if (f.factor.includes("anxiety")) return `${f.factor.toLowerCase()} (${f.value})`;
        if (f.factor === "Poor sleep quality") return `poor sleep quality (${f.value})`;
        if (f.factor === "Sedentary lifestyle") return `low physical activity (${f.value})`;
        return f.factor.toLowerCase();
    }).join(", ");

    const extra = factors.length > 3 ? `, and ${factors.length - 3} additional concern(s)` : "";
    const pct = (pred.probability * 100).toFixed(1);

    let insight = factors.length > 0
        ? `${pred.risk_label} (probability ${pct}%) due to ${causePhrase}${extra}.`
        : `${pred.risk_label} (probability ${pct}%). No major clinical risk factors were identified.`;

    insight += ` Among 5 similar patient profiles retrieved from the knowledge base, 3 (60%) had diagnosed heart disease -- above population baseline, warranting further investigation.`;

    const hasPhysical = factors.some(f => ["High cholesterol", "Hypertension", "Obesity"].includes(f.factor));
    const hasMental = factors.some(f => ["High stress", "Severe anxiety", "Poor sleep quality", "Sedentary lifestyle"].includes(f.factor));
    if (hasPhysical && hasMental) {
        insight += " Notably, this patient presents concurrent physical and mental health risk factors, which research shows can compound cardiovascular risk by 1.5-2x through stress-mediated inflammation and autonomic dysfunction.";
    }

    // Recommendations
    const recs = [];
    if (pred.probability > 0.7) recs.push("Urgent: Schedule comprehensive cardiovascular evaluation immediately.");
    factors.forEach(f => {
        if (f.factor === "High cholesterol") recs.push("Consult physician about cholesterol management (diet/statins).");
        if (f.factor === "Hypertension") recs.push("Monitor blood pressure daily. Consider antihypertensive therapy.");
        if (f.factor === "Obesity") recs.push("Aim for gradual weight loss (0.5-1 kg/week) through diet and exercise.");
        if (f.factor === "High stress") recs.push("Implement stress management: meditation, therapy, or structured relaxation.");
        if (f.factor.includes("anxiety")) recs.push("Refer to mental health professional for anxiety assessment (GAD-7 screening).");
        if (f.factor === "Poor sleep quality") recs.push("Address sleep hygiene. Consider sleep study if quality remains poor.");
        if (f.factor === "Sedentary lifestyle") recs.push("Start with 150 min/week moderate activity (walking, cycling).");
    });
    if (recs.length === 0) recs.push("Continue healthy lifestyle. Annual checkup recommended.");

    const similarPatients = [
        { patient_id: "P0155", similarity: 0.5139, risk: "High" },
        { patient_id: "P0285", similarity: 0.5113, risk: "High" },
        { patient_id: "P0100", similarity: 0.5109, risk: "Low" },
        { patient_id: "P0271", similarity: 0.5106, risk: "Low" },
        { patient_id: "P0073", similarity: 0.5104, risk: "High" },
    ];

    return {
        insight,
        ml_prediction: pred,
        rag_context: `Based on analysis of 5 similar patient profiles, 3/5 (60%) have heart disease. Key risk factors include: elevated BMI (30.5), significant anxiety (12/21). Average age in this cohort: 57 years. Recommend comprehensive cardiovascular screening and mental health assessment.`,
        similar_patients: similarPatients,
        recommendations: recs,
        risk_factors: factors,
    };
}

/* ═══════════════════════════════════════════════════════════════════
   API WRAPPER (auto-selects live API or demo engine)
   ═══════════════════════════════════════════════════════════════════ */

async function apiPredict(data) {
    if (DEMO_MODE) return demoPredict(data);
    const res = await fetch(`${API}/predict`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
}

async function apiAsk(query) {
    if (DEMO_MODE) return demoAsk(query);
    const res = await fetch(`${API}/ask`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ query, top_k: 5 }) });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
}

async function apiInsight(data) {
    if (DEMO_MODE) return demoInsight(data);
    const res = await fetch(`${API}/insight`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
}

/* ═══════════════════════════════════════════════════════════════════
   EVENT HANDLERS
   ═══════════════════════════════════════════════════════════════════ */

document.getElementById("predictBtn").innerHTML = BUTTON_LABELS.predict;
document.getElementById("insightBtn").innerHTML = BUTTON_LABELS.insight;

// ── Predict Form ─────────────────────────────────────────────────
document.getElementById("predictForm").addEventListener("submit", async e => {
    e.preventDefault();
    const btn = document.getElementById("predictBtn");
    btn.disabled = true;
    btn.innerHTML = BUTTON_LABELS.predicting;

    try {
        const data = getFormData();
        const result = await apiPredict(data);
        displayPrediction(result);
    } catch (err) {
        alert(`Prediction failed: ${err.message}`);
    } finally {
        btn.disabled = false;
        btn.innerHTML = BUTTON_LABELS.predict;
    }
});

// ── Chat Form ────────────────────────────────────────────────────
document.getElementById("chatForm").addEventListener("submit", async e => {
    e.preventDefault();
    const input = document.getElementById("chatInput");
    const query = input.value.trim();
    if (!query) return;
    input.value = "";
    await sendChat(query);
});

function askSuggestion(button) { sendChat(button.textContent); }

async function sendChat(query) {
    const messages = document.getElementById("chatMessages");
    messages.innerHTML += `<div class="chat-message user"><div class="message-avatar user-avatar">${icon("user")}</div><div class="message-content"><p>${escapeHtml(query)}</p></div></div>`;

    const loadId = `load-${Date.now()}`;
    messages.innerHTML += `<div class="chat-message bot" id="${loadId}"><div class="message-avatar assistant-avatar">${icon("brain")}</div><div class="message-content">${loadingHTML()}</div></div>`;
    messages.scrollTop = messages.scrollHeight;

    try {
        const data = await apiAsk(query);
        const sourceTags = data.sources.map(s =>
            `<span class="source-tag">${escapeHtml(s.patient_id)} (${escapeHtml(s.risk)}, ${(s.similarity * 100).toFixed(0)}%)</span>`
        ).join("");

        document.getElementById(loadId).querySelector(".message-content").innerHTML = `
            <p>${escapeHtml(data.response)}</p>
            <div class="source-tags">${sourceTags}</div>
        `;
    } catch (err) {
        document.getElementById(loadId).querySelector(".message-content").innerHTML =
            `<p style="color:var(--red);display:flex;align-items:center;gap:0.45rem;">${icon("alert")}<span>Error: ${escapeHtml(err.message)}</span></p>`;
    }
    messages.scrollTop = messages.scrollHeight;
}

// ── Insight Generation ───────────────────────────────────────────
async function generateInsight() {
    const btn = document.getElementById("insightBtn");
    btn.disabled = true;
    btn.innerHTML = BUTTON_LABELS.generating;

    try {
        const data = PROFILES[currentProfile];
        const result = await apiInsight(data);
        displayInsight(result);
    } catch (err) {
        alert(`Insight generation failed: ${err.message}`);
    } finally {
        btn.disabled = false;
        btn.innerHTML = BUTTON_LABELS.insight;
    }
}

/* ═══════════════════════════════════════════════════════════════════
   DISPLAY FUNCTIONS
   ═══════════════════════════════════════════════════════════════════ */

function displayPrediction(r) {
    document.getElementById("riskCard").style.display = "block";
    document.getElementById("factorsCard").style.display = "block";

    const pct = Math.round(r.probability * 100);
    const circumference = 2 * Math.PI * 54;
    const offset = circumference * (1 - r.probability);
    const fill = document.getElementById("gaugeFill");
    const color = getRiskColor(r.probability);

    fill.style.strokeDashoffset = offset;
    fill.style.stroke = color;

    document.getElementById("gaugeValue").textContent = `${pct}%`;
    document.getElementById("gaugeValue").style.color = color;
    document.getElementById("gaugeLabel").textContent = r.risk_level.toUpperCase();
    document.getElementById("riskLabel").textContent = r.risk_label;
    document.getElementById("riskLabel").style.color = color;
    document.getElementById("riskLevel").textContent = `${capitalize(r.risk_level)} risk level • ${Math.round(r.confidence * 100)}% confidence`;
    document.getElementById("modelUsed").textContent = `Model: ${r.model_used}`;

    const factorsList = document.getElementById("factorsList");
    if (r.contributing_factors && r.contributing_factors.length > 0) {
        factorsList.innerHTML = r.contributing_factors.map(factor => `
            <div class="factor-item ${factor.severity}">
                <span class="factor-badge ${factor.severity}">${factor.severity}</span>
                <div class="factor-info">
                    <div class="factor-name">${escapeHtml(factor.factor)} <span class="factor-value">${escapeHtml(String(factor.value))}</span></div>
                    <div class="factor-detail">${escapeHtml(factor.detail)}</div>
                </div>
            </div>
        `).join("");
    } else {
        factorsList.innerHTML = `<div class="empty-state">${icon("check")}<span>No significant risk factors identified.</span></div>`;
    }

    document.getElementById("riskCard").scrollIntoView({ behavior: "smooth", block: "start" });
}

function displayInsight(r) {
    document.getElementById("insightResults").style.display = "flex";

    const probability = r.ml_prediction.probability;
    const probabilityText = `${(probability * 100).toFixed(1)}%`;
    const color = getRiskColor(probability);

    document.getElementById("insightSummary").innerHTML = `
        <h3 class="panel-title">
            <span class="panel-icon">${icon("spark")}</span>
            Health Insight
        </h3>
        <p>${escapeHtml(r.insight)}</p>
        <div class="insight-metrics">
            <div class="metric">
                <span class="metric-label">Risk Probability</span>
                <span class="metric-value large" style="color:${color}">${probabilityText}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Risk Level</span>
                <span class="metric-value" style="color:${color}">${escapeHtml(r.ml_prediction.risk_label)}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Model</span>
                <span class="metric-value">${escapeHtml(r.ml_prediction.model_used)}</span>
            </div>
        </div>
    `;

    document.getElementById("ragContext").textContent = r.rag_context;
    document.getElementById("recsList").innerHTML = r.recommendations.map(rec => `<li>${escapeHtml(rec)}</li>`).join("");
    document.getElementById("similarList").innerHTML = r.similar_patients.map(patient => {
        const similarity = `${(patient.similarity * 100).toFixed(0)}%`;
        const riskClass = getRiskClass(patient.risk);
        return `
            <div class="similar-patient">
                <span class="patient-id">${escapeHtml(patient.patient_id)}</span>
                <span class="patient-risk ${riskClass}">${escapeHtml(patient.risk)} Risk</span>
                <div class="similarity-wrap">
                    <div class="similarity-bar"><div class="similarity-fill" style="width:${similarity}"></div></div>
                    <span class="patient-id">${similarity}</span>
                </div>
            </div>
        `;
    }).join("");

    document.getElementById("insightResults").scrollIntoView({ behavior: "smooth" });
}
