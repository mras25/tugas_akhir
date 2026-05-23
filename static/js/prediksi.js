// ============================================================
// prediksi.js — Halaman Prediksi & Evaluasi Model
// ============================================================

let activeLocation = null;
let predChart      = null;
let isLoading      = false;

// ---------- Muat daftar lokasi ----------
async function loadLocations() {
  const res  = await fetch("/api/locations");
  const data = await res.json();

  const select = document.getElementById("locationSelect");
  select.innerHTML = '<option value="">Pilih lokasi...</option>';

  (data.locations || []).forEach(loc => {
    const opt       = document.createElement("option");
    opt.value       = loc;
    opt.textContent = loc.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
    select.appendChild(opt);
  });
}

// ---------- Muat dropdown fitur ----------
async function loadFeatures(location) {
  if (!location) return;

  const res  = await fetch(`/api/features?location=${location}`);
  const data = await res.json();
  if (data.error) return;

  const select = document.getElementById("predFeatureSelect");
  select.innerHTML = '<option value="">Pilih fitur...</option>';

  (data.features || []).forEach(f => {
    const opt       = document.createElement("option");
    opt.value       = f;
    opt.textContent = f;
    select.appendChild(opt);
  });

  // Pilih elev_m secara default
  const elvOpt = [...select.options].find(o => o.value === "elev_m");
  if (elvOpt) {
    select.value = "elev_m";
  } else if (select.options.length > 1) {
    select.selectedIndex = 1;
  }
}

// ---------- Format tanggal ke dd-mm-yyyy HH:MM ----------
function formatLabel(raw) {
  if (!raw) return raw;
  const s = String(raw).trim();

  // Biarkan label +Nj apa adanya
  if (/^\+\d+j$/.test(s)) return s;

  // Format dari Python strftime: "28/07/2025 00:00" atau "28/07/2025 00:00:00"
  // new Date() gagal parse ini karena mengira bulan duluan → parse manual
  const dmyMatch = s.match(
    /^(\d{2})\/(\d{2})\/(\d{4})\s+(\d{2}):(\d{2})(?::\d{2})?$/
  );
  if (dmyMatch) {
    // [_, dd, mm, yyyy, HH, MM]
    return `${dmyMatch[1]}-${dmyMatch[2]}-${dmyMatch[3]} ${dmyMatch[4]}:${dmyMatch[5]}`;
  }

  // Format ISO "2025-07-28T00:00:00" atau "2025-07-28 00:00"
  const isoMatch = s.match(
    /^(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})/
  );
  if (isoMatch) {
    return `${isoMatch[3]}-${isoMatch[2]}-${isoMatch[1]} ${isoMatch[4]}:${isoMatch[5]}`;
  }

  // Fallback: coba new Date() untuk format lain
  const d = new Date(s);
  if (!isNaN(d.getTime())) {
    const dd   = String(d.getDate()).padStart(2, "0");
    const mm   = String(d.getMonth() + 1).padStart(2, "0");
    const yyyy = d.getFullYear();
    const HH   = String(d.getHours()).padStart(2, "0");
    const MM   = String(d.getMinutes()).padStart(2, "0");
    return `${dd}-${mm}-${yyyy} ${HH}:${MM}`;
  }

  // Tidak dikenali, kembalikan aslinya
  return s;
}

// ---------- Render grafik pred vs actual + forecast ----------
// Hanya membaca dari cache JSON yang sudah disimpan oleh admin (runAllPrediction).
// Fungsi ini TIDAK pernah memicu ulang model — tidak ada &t=Date.now() bust cache.
async function renderPredVsActual() {
  if (!activeLocation) return;

  const selectedFeature = document.getElementById("predFeatureSelect").value;
  if (!selectedFeature) return;

  const infoEl = document.getElementById("predictionStatus");

  // Ambil pred-vs-actual dari cache (tanpa cache-bust agar backend tidak re-run)
  const res  = await fetch(
    `/api/pred-vs-actual?location=${activeLocation}&feature=${selectedFeature}`
  );
  const data = await res.json();

  if (data.error) {
    if (infoEl && !infoEl.innerHTML.includes("✅") && !infoEl.innerHTML.includes("⏳")) {
      infoEl.innerHTML = `<span style="color:#888">ℹ️ ${data.error}</span>`;
    }
    return;
  }

  const predValues   = data.pred      || [];
  const actualValues = data.actual    || [];
  const timeLabels   = (data.time || []).map(formatLabel);
  const threshold    = data.threshold ?? null;

  // Ambil forecast dari cache
  let futureValues  = [];
  let forecastSteps = [];

  const fRes = await fetch(
    `/api/forecast-result?location=${activeLocation}&feature=${selectedFeature}`
  );
  if (fRes.ok) {
    const fData   = await fRes.json();
    futureValues  = fData.values || [];
    // Gunakan steps berisi timestamp nyata (dd-mm-yyyy HH:MM) jika tersedia
    const rawSteps =
      fData.steps && fData.steps.length === futureValues.length
        ? fData.steps
        : futureValues.map((_, i) => `+${i + 1}j`);
    forecastSteps = rawSteps.map(formatLabel);
  }

  // Gabungkan label waktu
  const allLabels = [...timeLabels, ...forecastSteps];

  // Dataset per garis
  const predLine     = [...predValues,  ...Array(futureValues.length).fill(null)];
  const actualLine   = [...actualValues,...Array(futureValues.length).fill(null)];
  const forecastLine = [...Array(predValues.length).fill(null), ...futureValues];
  const threshLine   = threshold !== null ? Array(allLabels.length).fill(threshold) : [];

  const ctx = document.getElementById("predCanvas");
  if (!ctx) {
    console.error("Canvas #predCanvas tidak ditemukan di HTML");
    return;
  }

  const allValues = [...predValues, ...actualValues, ...futureValues].filter(v => v !== null);
  const minVal    = allValues.length ? Math.min(...allValues) : 0;
  const maxVal    = allValues.length ? Math.max(...allValues) : 1;
  const pad       = (maxVal - minVal) * 0.2 || 0.1;

  if (predChart) {
    predChart.destroy();
    predChart = null;
  }

  const datasets = [
    {
      label: `Prediksi (${selectedFeature})`,
      data: predLine,
      borderColor: "#2563eb",
      borderWidth: 2,
      pointRadius: 0,
      tension: 0.3,
      spanGaps: false
    },
    {
      label: `Aktual (${selectedFeature})`,
      data: actualLine,
      borderColor: "#dc2626",
      borderWidth: 2,
      pointRadius: 0,
      tension: 0.3,
      spanGaps: false
    },
    {
      label: "Forecast 48 Jam",
      data: forecastLine,
      borderColor: "#16a34a",
      borderWidth: 2,
      borderDash: [6, 6],
      pointRadius: 0,
      tension: 0.3,
      spanGaps: false
    }
  ];

  if (threshLine.length) {
    datasets.push({
      label: `Threshold (${threshold.toFixed(3)} m)`,
      data: threshLine,
      borderColor: "rgba(255,0,0,0.7)",
      borderWidth: 1.5,
      borderDash: [4, 4],
      pointRadius: 0,
      tension: 0
    });
  }

  predChart = new Chart(ctx, {
    type: "line",
    data: { labels: allLabels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      scales: {
        x: {
          ticks: { maxTicksLimit: 12, maxRotation: 45 }
        },
        y: {
          min: minVal - pad,
          max: maxVal + pad,
          title: { display: true, text: selectedFeature }
        }
      },
      plugins: {
        legend: { position: "top" },
        tooltip: { mode: "index", intersect: false }
      },
      interaction: { mode: "nearest", axis: "x", intersect: false }
    }
  });

  renderPredExplanation(predValues, actualValues, futureValues, threshold, selectedFeature);
}

// ---------- Penjelasan teks di bawah grafik ----------
function renderPredExplanation(pred, actual, forecast, threshold, featureName = "elev_m") {
  const box = document.getElementById("predExplanation");
  if (!box) return;

  if (!pred.length || !actual.length) {
    box.innerHTML = `<span style="font-size:13px;color:#888;">Belum ada data untuk ditampilkan.</span>`;
    return;
  }

  // RMSE dihitung dari prediksi historis vs aktual
  const n = Math.min(pred.length, actual.length);
  let sumSq = 0;
  for (let i = 0; i < n; i++) sumSq += (pred[i] - actual[i]) ** 2;
  const rmse = Math.sqrt(sumSq / n);

  // Potensi banjir dihitung dari data FORECAST (48 jam ke depan)
  const forecastData = forecast && forecast.length ? forecast : [];
  const floodCount   = forecastData.filter(v => threshold !== null && v >= threshold).length;
  const total        = forecastData.length;
  const pct          = total > 0 ? ((floodCount / total) * 100).toFixed(1) : "0.0";
  const isRisky      = floodCount > 0;
  const noForecast   = total === 0;

  box.innerHTML = `
    <div style="margin-top:14px;display:flex;gap:10px;flex-wrap:wrap;">
      <div style="flex:1;min-width:160px;padding:12px 16px;background:#f7f8fa;border:1px solid #e2e5ea;border-radius:8px;">
        <div style="font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.06em;color:#888;margin-bottom:4px;">RMSE Estimasi (${featureName})</div>
        <div style="font-size:22px;font-weight:700;color:#1a1a2e;font-family:monospace;">${rmse.toFixed(4)}</div>
        <div style="font-size:11px;color:#aaa;margin-top:3px;">Dari prediksi vs aktual historis</div>
      </div>
      <div style="flex:1;min-width:200px;padding:12px 16px;background:${noForecast ? "#f7f8fa" : isRisky ? "#fff5f5" : "#f0fdf4"};border:1px solid ${noForecast ? "#e2e5ea" : isRisky ? "#fecaca" : "#bbf7d0"};border-radius:8px;">
        <div style="font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.06em;color:${noForecast ? "#888" : isRisky ? "#c0392b" : "#27ae60"};margin-bottom:4px;">Potensi Melebihi Threshold (${threshold !== null ? threshold.toFixed(3) : "N/A"} m)</div>
        ${noForecast
          ? `<div style="font-size:14px;color:#aaa;">Data forecast belum tersedia</div>`
          : `<div style="font-size:22px;font-weight:700;color:${isRisky ? "#c0392b" : "#27ae60"};font-family:monospace;">${floodCount}<span style="font-size:13px;font-weight:400;"> dari ${total} langkah forecast (${pct}%)</span></div>`
        }
        <div style="font-size:11px;color:#aaa;margin-top:3px;">Berdasarkan forecast 48 jam ke depan</div>
      </div>
    </div>
  `;
}

// ---------- Jalankan semua prediksi (admin only) ----------
async function runAllPrediction() {
  if (isLoading) return;
  isLoading = true;

  const statusEl = document.getElementById("predictionStatus");
  const btn      = document.getElementById("runPrediksiBtn");

  statusEl.style.color = "#555";
  if (btn) btn.disabled = true;

  let dots = 0;
  const interval = setInterval(() => {
    dots = (dots + 1) % 4;
    statusEl.innerHTML = "⏳ Menjalankan prediksi" + ".".repeat(dots);
  }, 500);

  try {
    const location = document.getElementById("locationSelect").value;
    if (!location) {
      clearInterval(interval);
      statusEl.innerHTML   = "❌ Pilih lokasi terlebih dahulu";
      statusEl.style.color = "red";
      isLoading = false;
      if (btn) btn.disabled = false;
      return;
    }

    activeLocation = location;

    // 1. Muat fitur
    const featRes  = await fetch(`/api/features?location=${activeLocation}`);
    const featData = await featRes.json();
    const featureList = featData.features || [];

    // 2. Jalankan pred-vs-actual untuk setiap fitur
    //    (admin → backend menjalankan model & menyimpan cache)
    //    Cache-bust dengan &run=1 agar backend tahu ini permintaan eksplisit dari admin
    for (const f of featureList) {
      const pvaRes = await fetch(
        `/api/pred-vs-actual?location=${activeLocation}&feature=${f}&run=1&t=${Date.now()}`
      );
      if (!pvaRes.ok) {
        const err = await pvaRes.json();
        console.warn(`pred-vs-actual gagal untuk ${f}:`, err.error);
      }
    }

    // 3. Jalankan forecast untuk setiap fitur (timestamp nyata disimpan di cache)
    for (const f of featureList) {
      await fetch("/api/forecast", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ location: activeLocation, feature: f })
      });
    }

    // 4. Render dari cache
    await renderPredVsActual();

    clearInterval(interval);
    statusEl.innerHTML   = "✅ Prediksi selesai";
    statusEl.style.color = "green";

  } catch (err) {
    clearInterval(interval);
    statusEl.innerHTML   = `❌ Error: ${err.message}`;
    statusEl.style.color = "red";
    console.error(err);
  } finally {
    isLoading = false;
    if (btn) btn.disabled = false;
  }
}

// ---------- Admin Modal ----------
function adminLogin() {
  document.getElementById("adminModal").style.display = "flex";
}

function closeAdminModal() {
  document.getElementById("adminModal").style.display = "none";
}

async function submitAdminLogin() {
  const password = document.getElementById("adminPassword").value;
  const errorEl  = document.getElementById("loginError");
  errorEl.innerText = "";

  try {
    const res  = await fetch("/admin/login", {
      method: "POST",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ password })
    });
    const data = await res.json();

    if (res.ok && data.status === "ok") {
      closeAdminModal();
      location.reload();
    } else {
      errorEl.innerText = data.error || "Password salah!";
    }
  } catch (err) {
    console.error(err);
    errorEl.innerText = "Terjadi kesalahan!";
  }
}

async function adminLogout() {
  await fetch("/admin/logout", { credentials: "include" });
  location.reload();
}

// ---------- Expose ke HTML ----------
window.adminLogin       = adminLogin;
window.closeAdminModal  = closeAdminModal;
window.submitAdminLogin = submitAdminLogin;
window.adminLogout      = adminLogout;
window.runAllPrediction = runAllPrediction;

// ---------- Init ----------
// CATATAN: logika show/hide .admin-only dan teks status
// sepenuhnya diurus oleh updateAdminUI() di prediksi.html
// agar tidak ada konflik dua DOMContentLoaded.
document.addEventListener("DOMContentLoaded", async () => {
  await loadLocations();

  const locationSelect = document.getElementById("locationSelect");
  const featureSelect  = document.getElementById("predFeatureSelect");
  const runBtn         = document.getElementById("runPrediksiBtn");

  if (runBtn) {
    runBtn.addEventListener("click", runAllPrediction);
  }

  locationSelect.addEventListener("change", async () => {
    activeLocation = locationSelect.value;
    if (!activeLocation) return;
    await loadFeatures(activeLocation);
    // Hanya tampilkan cache yang sudah ada — tidak re-run model
    await renderPredVsActual();
  });

  featureSelect.addEventListener("change", () => {
    // Hanya render ulang dari cache — TIDAK memicu ulang model
    renderPredVsActual();
  });
});
