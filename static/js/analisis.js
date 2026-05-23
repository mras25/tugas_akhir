Chart.register(window['chartjs-chart-matrix']);


  function adminLogin(){
    document.getElementById("adminModal").style.display = "flex";
  }

  function closeAdminModal(){
    document.getElementById("adminModal").style.display = "none";
  }

  async function submitAdminLogin(){

    const password = document.getElementById("adminPassword").value;
    const errorText = document.getElementById("loginError");

    const res = await fetch("/admin/login", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ password })
    });

    const data = await res.json();

    if(res.ok){
      closeAdminModal();
      location.reload();
    }else{
      errorText.innerText = data.error || "Login gagal";
    }
  }

  window.adminLogin = adminLogin;

  async function adminLogout(){
    await fetch("/admin/logout");
    location.reload();
  }

  window.adminLogout = adminLogout;


document.addEventListener("DOMContentLoaded", async () => {

  
  
  
  const uploadForm = document.getElementById("uploadForm");
  const uploadStatus = document.getElementById("uploadStatus");
  const evalBtn = document.getElementById("evalBtn");
  const evalStatus = document.getElementById("evalStatus");
  const featureSelect = document.getElementById("featureSelect");
  const validationBox = document.getElementById("validationBox");
  const validationList = document.getElementById("validationList");
  const locationSelect = document.getElementById("locationSelect");
  const fileInput = document.getElementById("csvFile");

  await loadLocations();


  let activeLocation = null;
  let timeSeriesChart = null;
  let evalChart = null;






  
  
  
  async function loadLocations() {
    const res = await fetch("/api/locations");
    const data = await res.json();

    locationSelect.innerHTML =
      `<option value="">Pilih Lokasi...</option>`;

    
    data.locations.forEach(loc => {
      const opt = document.createElement("option");
      opt.value = loc;
      opt.textContent = loc.replaceAll("_", " "); 
      locationSelect.appendChild(opt);
    });


    
    if (data.locations.length > 0) {

      if (featureSelect.options.length > 1) {
        featureSelect.selectedIndex = 1;
      }
    }
  }


  
  
  
  uploadForm.addEventListener("submit", async (e) => {
    e.preventDefault();

    if (!fileInput.files.length) {
      uploadStatus.innerText = "❌ Pilih file CSV terlebih dahulu";
      return;
    }

    const locationSelectUpload = document.getElementById("uploadLocationSelect");
    const location = locationSelectUpload.value;

    if (!location) {
      uploadStatus.innerText = "❌ Isi nama lokasi terlebih dahulu";
      return;
    }

    activeLocation = location;

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);
    formData.append("location", location);

    uploadStatus.innerText = "⏳ Mengunggah dataset...";

    const res = await fetch("/api/upload", {
      method: "POST",
      body: formData
      
    });

    console.log("STATUS:", res.status);

    if (!res.ok) {
      const data = await res.json();
      uploadStatus.innerText = "❌ " + (data.error || "Upload gagal");
      return;
    }

    const data = await res.json();

    if (data.status === "ok") {

      uploadStatus.innerText =
        `✅ Dataset berhasil diupload (${data.rows} baris)`;

      evalBtn.disabled = false;

      await loadLocations();

      
      locationSelect.value = location;
      activeLocation = location;

      await loadFeatureDropdown();

      if (featureSelect.options.length > 1) {
        featureSelect.selectedIndex = 1;
        loadTimeSeries(featureSelect.value);
      }

      renderClusterVisualization();   // ✅ WAJIB
      loadEvaluationForUser();        // (biar konsisten)
      renderValidation(data.validation);

    } else {
      uploadStatus.innerText = `❌ ${data.error}`;
    }
  });


  
  
  
  async function loadFeatureDropdown() {

    if (!activeLocation) return;

    const res = await fetch(`/api/features?location=${activeLocation}`);
    const data = await res.json();

    if (data.error) return;

    featureSelect.innerHTML =
      `<option value="">Pilih Fitur Time Series...</option>`;

    data.features.forEach(f => {
      const opt = document.createElement("option");
      opt.value = f;
      opt.textContent = f;
      featureSelect.appendChild(opt);
    });
  }


  
  
  
  async function loadTimeSeries(feature = null) {
    if (!activeLocation) return;

    let url = `/api/timeseries?location=${activeLocation}`;
    if (feature) url += `&feature=${feature}`;

    const res = await fetch(url);
    const data = await res.json();
    if (data.error) return;

    const container = document.getElementById("timeSeriesGraph");
    container.innerHTML = "<canvas></canvas>";
    const ctx = container.querySelector("canvas");

    if (timeSeriesChart) timeSeriesChart.destroy();

    timeSeriesChart = new Chart(ctx, {
      type: "line",
      data: {
        labels: data.time,
        datasets: [{
          label: `${data.feature} (${data.location})`,
          data: data.values,
          borderWidth: 2,
          tension: 0.3
        }]
      },
      options: {
        responsive: true,
        plugins: {
          legend: { display: true }
        }
      }
    });
  }

  featureSelect.addEventListener("change", () => {
    if (featureSelect.value) {
      loadTimeSeries(featureSelect.value);
    }
  });

  locationSelect.addEventListener("change", async () => {
    
    activeLocation = locationSelect.value;

    await loadFeatureDropdown();
    
    
    if (featureSelect.options.length > 1) {
      featureSelect.selectedIndex = 1;
      loadTimeSeries(featureSelect.value);
    }

    await loadEvaluationForUser();
    await renderClusterVisualization();
    
    


  });

  
  
  
  async function evaluateModel() {

    const res = await fetch(`/api/evaluate`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ location: activeLocation })
    });

    const data = await res.json();

    evalStatus.innerHTML = `
      RMSE: ${data.rmse}<br>
      MAE: ${data.mae}
    `;

    renderEvaluationChart(data.rmse, data.mae);
  }

  window.evaluateModel = evaluateModel;

  function renderForecastTable(steps, elev_m, status){

    let html = `<table>
      <tr>
        <th>Waktu</th>
        <th>elev_m (m)</th>
        <th>Status</th>
      </tr>`;

    for(let i=0;i<steps.length;i++){
      html += `
        <tr>
          <td>${steps[i]}</td>
          <td>${elev_m[i].toFixed(2)}</td>
          <td>${status[i] ? "POTENSI BANJIR" : "MINIM POTENSI BANJIR"}</td>
        </tr>
      `;
    }

    html += "</table>";

    document.getElementById("forecastTableContainer").innerHTML = html;
  }

  
  
  function renderEvaluationChart(rmse, mae) {

    const container = document.getElementById("evaluationGraph");
    container.innerHTML = "<canvas></canvas>";
    const ctx = container.querySelector("canvas");

    new Chart(ctx, {
      type: "bar",
      data: {
        labels: ["RMSE", "MAE"],
        datasets: [{
          data: [rmse, mae],
          borderWidth: 1
        }]
      }
    });
  }

  function renderEvalExplanation(rmse, mae){

  let level, levelColor, levelBg, levelBorder, icon;

  if(rmse < 0.1){
    level = "Sangat Akurat"; icon = "✦";
    levelColor = "#27ae60"; levelBg = "#f0fdf4"; levelBorder = "#bbf7d0";
  } else if(rmse < 0.3){
    level = "Cukup Baik"; icon = "◈";
    levelColor = "#d97706"; levelBg = "#fffbeb"; levelBorder = "#fde68a";
  } else {
    level = "Perlu Perbaikan"; icon = "⚠";
    levelColor = "#c0392b"; levelBg = "#fff5f5"; levelBorder = "#fecaca";
  }

  document.getElementById("evalExplanation").innerHTML = `
    <div style="margin-top:14px;display:flex;gap:10px;flex-wrap:wrap;">
      <div style="flex:1;min-width:140px;padding:14px 18px;background:#f7f8fa;border:1px solid #e2e5ea;border-radius:8px;">
        <div style="font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.07em;color:#888;margin-bottom:6px;">RMSE</div>
        <div style="font-size:24px;font-weight:700;color:#1a1a2e;font-family:monospace;line-height:1;">${rmse.toFixed(4)}</div>
        <div style="font-size:11px;color:#aaa;margin-top:4px;">Kesalahan rata-rata kuadrat</div>
      </div>
      <div style="flex:1;min-width:140px;padding:14px 18px;background:#f7f8fa;border:1px solid #e2e5ea;border-radius:8px;">
        <div style="font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.07em;color:#888;margin-bottom:6px;">MAE</div>
        <div style="font-size:24px;font-weight:700;color:#1a1a2e;font-family:monospace;line-height:1;">${mae.toFixed(4)}</div>
        <div style="font-size:11px;color:#aaa;margin-top:4px;">Kesalahan rata-rata absolut</div>
      </div>
      <div style="flex:1;min-width:160px;padding:14px 18px;background:${levelBg};border:1px solid ${levelBorder};border-radius:8px;">
        <div style="font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.07em;color:#888;margin-bottom:6px;">Kualitas Model</div>
        <div style="font-size:18px;font-weight:700;color:${levelColor};line-height:1;">${icon} ${level}</div>
        <div style="font-size:11px;color:#aaa;margin-top:4px;">Berdasarkan nilai RMSE</div>
      </div>
    </div>
  `;
}


  
  
  function renderValidation(validation) {
    validationList.innerHTML = "";
    validationBox.style.display = "block";

    if (
      validation.missing_features.length === 0 &&
      validation.ignored_features.length === 0
    ) {
      validationList.innerHTML =
        "<li style='color:green;'>✅ Semua fitur sesuai dengan model</li>";
      return;
    }

    validation.missing_features.forEach(f => {
      const li = document.createElement("li");
      li.style.color = "red";
      li.textContent = `❌ Fitur hilang: ${f}`;
      validationList.appendChild(li);
    });

    validation.ignored_features.forEach(f => {
      const li = document.createElement("li");
      li.style.color = "orange";
      li.textContent = `⚠️ Fitur diabaikan: ${f}`;
      validationList.appendChild(li);
    });
  }

async function loadEvaluationForUser(){

  if (!activeLocation) return;

  const res = await fetch(`/api/evaluation-result?location=${activeLocation}`);
  const data = await res.json();

  if (data.error) return;

  renderEvaluationChart(data.rmse, data.mae);
  renderEvalExplanation(data.rmse, data.mae);
  renderSummary(data.rmse, data.cluster_table);

}

      
function renderSummary(rmse, table){

  if(!table || table.length === 0){
    document.getElementById("summaryBox").innerHTML = "";
    return;
  }

  const values = table.map(r => r.mean);
  const maxMean = Math.max(...values);
  const isRisky = maxMean > 0;
  const riskColor  = isRisky ? "#c0392b" : "#27ae60";
  const riskBg     = isRisky ? "#fff5f5" : "#f0fdf4";
  const riskBorder = isRisky ? "#fecaca" : "#bbf7d0";
  const riskText   = isRisky ? "⚠ Potensi banjir terdeteksi pada kondisi pasang tinggi." : "✔ Kondisi saat ini minim potensi banjir.";

  document.getElementById("summaryBox").innerHTML = `
    <div style="margin-top:12px;padding:14px 18px;background:${riskBg};border:1px solid ${riskBorder};border-radius:8px;font-size:13px;line-height:1.7;">
      <div style="font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.07em;color:#888;margin-bottom:8px;">Kesimpulan Analisis</div>
      <div style="display:flex;flex-direction:column;gap:4px;color:#444;">
        <span>Model memiliki RMSE sebesar <b style="color:#1a1a2e;font-family:monospace;">${rmse.toFixed(3)}</b></span>
        <span>Nilai rata-rata tertinggi antar cluster: <b style="color:#1a1a2e;font-family:monospace;">${maxMean.toFixed(3)} m</b></span>
        <span style="font-weight:600;color:${riskColor};margin-top:4px;">${riskText}</span>
      </div>
    </div>
  `;
}
  
async function renderClusterVisualization() {

  if (!activeLocation) return;

  const res = await fetch(`/api/cluster-visual?location=${activeLocation}`);
  const data = await res.json();

  console.log("CLUSTER DATA:", data); // 🔥 taruh di sini

  if (data.error) return;

  const container = document.getElementById("clusterGraph");
  container.innerHTML = "<canvas></canvas>";
  const ctx = container.querySelector("canvas");

  const clusters = data.cluster_table.map(row => row.Cluster);
  const elev_mValues = data.cluster_table.map(row => row.mean);

  new Chart(ctx, {
    type: "bar",
    data: {
      labels: clusters,
      datasets: [{
        label: "Rata-rata elev_m per Cluster",
        data: elev_mValues,
        backgroundColor: elev_mValues.map(v => 
          v > 0.2 ? "rgba(255,0,0,0.6)" : "rgba(0,150,0,0.6)"
        )
      }]
    },
      options: {
        responsive: true,
        plugins: {
          annotation: {
            annotations: {
              line1: {
                type: 'line',
                yMin: 0.2,
                yMax: 0.2,
                borderColor: 'black',
                borderWidth: 2,
                label: {
                  content: 'Threshold Banjir',
                  enabled: true
                }
              }
            }
          }
        }
      }
    });

  renderClusterTable(data.cluster_table);
  renderClusterExplanation(data.cluster_table);
  await loadEvaluationForUser(); // Pastikan evaluasi juga terupdate setelah visualisasi cluster
}

  function renderClusterTable(table){

    const container = document.getElementById("clusterTableContainer");

    if(!table || table.length === 0){
      container.innerHTML = "Tidak ada data cluster";
      return;
    }

    // Tentukan cluster mana yang "banjir" berdasarkan mean tertinggi
    const maxMean = Math.max(...table.map(r => r.mean));

    // Pemetaan nama kolom teknis ke bahasa awam
    const labelMap = {
      Cluster: "Status",
      count:   "Jumlah Data",
      max:     "Nilai Tertinggi (m)",
      mean:    "Rata-rata (m)",
      min:     "Nilai Terendah (m)"
    };

    // Susun sebagai kartu vertikal, satu kartu per cluster, di tengah
    let html = '<div style="display:flex;flex-direction:column;gap:12px;max-width:480px;margin:16px auto 0;">';

    table.forEach((row) => {
      const isFlood = row.mean === maxMean;
      const badgeLabel  = isFlood ? "🔴 Potensi Banjir" : "🟢 Minim Potensi Banjir";
      const badgeColor  = isFlood ? "#c0392b"   : "#27ae60";
      const badgeBg     = isFlood ? "#fff0f0"   : "#f0fdf4";
      const badgeBorder = isFlood ? "#fecaca"   : "#bbf7d0";
      const cardBorder  = isFlood ? "#fecaca"   : "#bbf7d0";
      const cardBg      = isFlood ? "#fffafa"   : "#f9fffe";

      html += `
        <div style="border:1px solid ${cardBorder};border-radius:10px;background:${cardBg};padding:16px 20px;">
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px;">
            <span style="display:inline-block;padding:4px 14px;border-radius:999px;font-size:13px;font-weight:700;color:${badgeColor};background:${badgeBg};border:1px solid ${badgeBorder};">${badgeLabel}</span>
          </div>
          <table style="width:100%;border-collapse:collapse;font-size:13px;">
            <tbody>`;

      const rowDefs = [
        { key: "count", label: "Jumlah Data",        fmt: v => Math.round(v).toLocaleString("id-ID") },
        { key: "mean",  label: "Rata-rata (m)",       fmt: v => Number(v).toFixed(3) },
        { key: "max",   label: "Nilai Tertinggi (m)", fmt: v => Number(v).toFixed(3) },
        { key: "min",   label: "Nilai Terendah (m)",  fmt: v => Number(v).toFixed(3) },
      ];

      rowDefs.forEach(({ key, label, fmt }) => {
        if(row[key] === undefined) return;
        const isVal  = key === "mean";
        const valColor = isVal ? badgeColor : "#333";
        const valWeight = isVal ? "700" : "400";
        html += `
              <tr>
                <td style="padding:6px 0;color:#888;font-size:12px;width:55%;">${label}</td>
                <td style="padding:6px 0;color:${valColor};font-weight:${valWeight};text-align:right;">${fmt(row[key])}</td>
              </tr>`;
      });

      html += `
            </tbody>
          </table>
        </div>`;
    });

    html += '</div>';
    container.innerHTML = html;
  }

  

});

function renderClusterExplanation(table){

  const elev = table.map(r => r.mean);
  const min = Math.min(...elev);
  const max = Math.max(...elev);

  document.getElementById("clusterExplanation").innerHTML = `
    <div style="margin-top:12px;padding:12px 16px;background:#f7f8fa;border-left:3px solid #3b82f6;border-radius:0 6px 6px 0;font-size:13px;line-height:1.7;">
      <span style="display:block;font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:#555;margin-bottom:6px;">Interpretasi Cluster</span>
      <span style="display:block;color:#27ae60;">✔ Cluster rendah (~${min.toFixed(2)} m) — kondisi minim potensi banjir</span>
      <span style="display:block;color:#c0392b;">⚠ Cluster tinggi (~${max.toFixed(2)} m) — potensi banjir</span>
    </div>
  `;
}



window.adminLogin = adminLogin;
