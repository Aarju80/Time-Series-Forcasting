let chart = null;
let backtestChart = null;

/* ========== CHART.JS GLOBAL DARK THEME ========== */
Chart.defaults.color = "#b0b8cc";
Chart.defaults.borderColor = "rgba(80, 100, 140, 0.2)";
Chart.defaults.plugins.legend.labels.usePointStyle = true;
Chart.defaults.plugins.legend.labels.pointStyleWidth = 10;
Chart.defaults.plugins.legend.labels.padding = 16;
Chart.defaults.font.family = "'Inter', sans-serif";

/* ========== STATUS ========== */
function setStatus(msg) {
    document.getElementById("status").innerText = msg;

    const badge = document.getElementById("statusBadge");
    if (msg.includes("⏳") || msg.includes("Training")) {
        badge.classList.add("training-pulse");
    } else {
        badge.classList.remove("training-pulse");
    }
}

/* ========== UPLOAD ========== */
async function upload() {
    const fileInput = document.getElementById("csvFile");

    if (!fileInput.files.length) {
        setStatus("❌ Please choose a CSV file.");
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    try {
        await fetch("http://127.0.0.1:8000/upload", {
            method: "POST",
            body: formData
        });

        document.getElementById("trainBtn").disabled = false;
        setStatus("✅ File uploaded. Click Train Model.");
    } catch (err) {
        console.error(err);
        setStatus("❌ Upload failed. Is the backend running?");
    }
}

/* ========== TRAIN ========== */
async function train() {
    setStatus("⏳ Training started (this may take a minute)...");
    document.getElementById("trainBtn").disabled = true;

    try {
        await fetch("http://127.0.0.1:8000/train", { method: "POST" });
        pollStatus();
    } catch (err) {
        console.error(err);
        setStatus("❌ Training request failed.");
    }
}

async function pollStatus() {
    try {
        const res = await fetch("http://127.0.0.1:8000/status");
        const data = await res.json();

        setStatus(data.message);

        if (data.status === "training") {
            setTimeout(pollStatus, 2000);
        }

        if (data.status === "done") {
            document.getElementById("predictBtn").disabled = false;
            setStatus("✅ Training complete. Running backtest...");
            runBacktest();
        }
    } catch (err) {
        console.error(err);
        setStatus("❌ Status check failed.");
    }
}

/* ========== PREDICT ========== */
async function predict() {
    const horizon = document.getElementById("horizon").value;

    try {
        const res = await fetch(
            `http://127.0.0.1:8000/predict?horizon=${horizon}`,
            { method: "POST" }
        );

        if (!res.ok) {
            const err = await res.text();
            setStatus("❌ " + err);
            return;
        }

        const data = await res.json();
        drawChart(data.columns, data.predictions);
        setStatus(`✅ Showing ${horizon}-day forecast`);
    } catch (err) {
        console.error(err);
        setStatus("❌ Prediction request failed.");
    }
}

/* ========== BACKTEST ========== */
async function runBacktest() {
    try {
        const res = await fetch("http://127.0.0.1:8000/backtest", {
            method: "POST"
        });

        if (!res.ok) {
            console.error("Backtest failed:", await res.text());
            setStatus("✅ Training complete. Click Predict.");
            return;
        }

        const data = await res.json();
        drawBacktestChart(data);
        showMetrics(data.metrics);
        setStatus("✅ Model ready — backtest results below.");
    } catch (err) {
        console.error("Backtest error:", err);
        setStatus("✅ Training complete. Click Predict.");
    }
}

/* ========== METRICS ========== */
function showMetrics(metrics) {
    const section = document.getElementById("metricsSection");
    const container = document.getElementById("metricsCards");

    section.classList.remove("hidden");
    container.innerHTML = "";

    const priorityCols = ["close", "adj close", "open", "high", "low"];
    const cols = Object.keys(metrics);
    const displayCols = priorityCols.filter(c => cols.includes(c));
    if (displayCols.length === 0) displayCols.push(cols[0]);

    for (const col of displayCols) {
        const m = metrics[col];

        let mapeClass = "color-green";
        if (m.mape > 5) mapeClass = "color-amber";
        if (m.mape > 10) mapeClass = "color-red";

        const card = document.createElement("div");
        card.className = "metric-card";
        card.innerHTML = `
            <div class="metric-col-name">${col}</div>
            <div class="metric-grid">
                <div>
                    <div class="metric-value big ${mapeClass}">${m.mape}%</div>
                    <div class="metric-label">MAPE</div>
                </div>
                <div>
                    <div class="metric-value med color-blue">${formatNum(m.mae)}</div>
                    <div class="metric-label">MAE</div>
                </div>
                <div>
                    <div class="metric-value med color-purple">${formatNum(m.rmse)}</div>
                    <div class="metric-label">RMSE</div>
                </div>
            </div>
        `;
        container.appendChild(card);
    }
}

function formatNum(n) {
    if (n >= 1000000) return (n / 1000000).toFixed(1) + "M";
    if (n >= 1000) return (n / 1000).toFixed(1) + "K";
    return n.toFixed(2);
}

/* ========== FORECAST CHART ========== */
function drawChart(columns, predictions) {
    const ctx = document.getElementById("chart").getContext("2d");
    if (chart) chart.destroy();

    const labels = predictions.map((_, i) => `T+${i + 1}`);

    const palette = [
        { line: "#5b8def", fill: "rgba(91, 141, 239, 0.08)" },
        { line: "#f97316", fill: "rgba(249, 115, 22, 0.08)" },
        { line: "#facc15", fill: "rgba(250, 204, 21, 0.08)" },
        { line: "#34d399", fill: "rgba(52, 211, 153, 0.08)" },
        { line: "#a78bfa", fill: "rgba(167, 139, 250, 0.08)" },
        { line: "#22d3ee", fill: "rgba(34, 211, 238, 0.08)" },
    ];

    const datasets = columns.map((col, idx) => {
        const isVolume = col.toLowerCase().includes("volume");
        const c = palette[idx % palette.length];

        return {
            label: col,
            data: predictions.map(p => p[idx]),
            borderColor: c.line,
            backgroundColor: c.fill,
            borderWidth: 2,
            tension: 0.4,
            pointRadius: 3,
            pointHoverRadius: 6,
            pointBackgroundColor: c.line,
            pointBorderColor: "transparent",
            fill: idx === 0,
            yAxisID: isVolume ? "y1" : "y"
        };
    });

    chart = new Chart(ctx, {
        type: "line",
        data: { labels, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: "index", intersect: false },
            plugins: {
                legend: { position: "top" },
                tooltip: {
                    backgroundColor: "rgba(10, 15, 30, 0.95)",
                    titleColor: "#f0f2f7",
                    bodyColor: "#b0b8cc",
                    borderColor: "rgba(80, 100, 140, 0.3)",
                    borderWidth: 1,
                    padding: 12,
                    cornerRadius: 8,
                }
            },
            scales: {
                x: { grid: { color: "rgba(80, 100, 140, 0.12)" } },
                y: {
                    type: "linear", position: "left",
                    title: { display: true, text: "Price", color: "#b0b8cc" },
                    grid: { color: "rgba(80, 100, 140, 0.12)" },
                },
                y1: {
                    type: "linear", position: "right",
                    title: { display: true, text: "Volume", color: "#b0b8cc" },
                    grid: { drawOnChartArea: false },
                }
            }
        }
    });
}

/* ========== BACKTEST CHART ========== */
function drawBacktestChart(data) {
    const section = document.getElementById("backtestSection");
    section.classList.remove("hidden");

    const ctx = document.getElementById("backtestChart").getContext("2d");
    if (backtestChart) backtestChart.destroy();

    let colIdx = 0;
    let colName = data.columns[0];
    for (let i = 0; i < data.columns.length; i++) {
        if (data.columns[i] === "close") {
            colIdx = i;
            colName = "close";
            break;
        }
    }

    const actuals = data.actuals.map(row => row[colIdx]);
    const preds = data.predictions.map(row => row[colIdx]);
    const labels = actuals.map((_, i) => `${i + 1}`);

    document.getElementById("backtestLabel").textContent =
        `${data.test_size} test points`;

    backtestChart = new Chart(ctx, {
        type: "line",
        data: {
            labels,
            datasets: [
                {
                    label: `Actual (${colName})`,
                    data: actuals,
                    borderColor: "#34d399",
                    backgroundColor: "rgba(52, 211, 153, 0.06)",
                    borderWidth: 2,
                    tension: 0.3,
                    pointRadius: 0,
                    fill: true,
                },
                {
                    label: `Predicted (${colName})`,
                    data: preds,
                    borderColor: "#f87171",
                    backgroundColor: "rgba(248, 113, 113, 0.06)",
                    borderWidth: 2,
                    borderDash: [6, 4],
                    tension: 0.3,
                    pointRadius: 0,
                    fill: false,
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: "index", intersect: false },
            plugins: {
                legend: { position: "top" },
                tooltip: {
                    backgroundColor: "rgba(10, 15, 30, 0.95)",
                    titleColor: "#f0f2f7",
                    bodyColor: "#b0b8cc",
                    borderColor: "rgba(80, 100, 140, 0.3)",
                    borderWidth: 1,
                    padding: 12,
                    cornerRadius: 8,
                }
            },
            scales: {
                x: {
                    title: { display: true, text: "Test Data Point", color: "#b0b8cc" },
                    ticks: { maxTicksLimit: 20 },
                    grid: { color: "rgba(80, 100, 140, 0.12)" },
                },
                y: {
                    title: {
                        display: true,
                        text: colName.charAt(0).toUpperCase() + colName.slice(1) + " Price",
                        color: "#b0b8cc",
                    },
                    grid: { color: "rgba(80, 100, 140, 0.12)" },
                }
            }
        }
    });
}
