let locations = [];
let alertHistory = [];
let indexLocation = 0;

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

async function adminLogout(){
  await fetch("/admin/logout");
  location.reload();
}

function formatNama(text){
  return text
    .replaceAll("_", " ")
    .replace(/\b\w/g, c => c.toUpperCase());
}

async function loadLocations(){

  const res = await fetch("/api/locations");
  const data = await res.json();

  locations = data.locations;

  if(locations.length === 0){
    document.getElementById("statusBanner").innerText =
      "Tidak ada dataset tersedia";
    return;
  }

  updateDashboard();

  setInterval(updateDashboard,5000); // ganti lokasi setiap 5 detik
}


async function updateDashboard(){

  const location = locations[indexLocation];

  // Ambil data dashboard (elev_m aktual) dan forecast sekaligus
  const [dashRes, forecastRes, evalRes] = await Promise.all([
    fetch(`/api/dashboard?location=${location}`),
    fetch(`/api/forecast-result?location=${location}&feature=elev_m`),
    fetch(`/api/evaluation-result?location=${location}`)
  ]);

  const data      = await dashRes.json();
  const evalData  = await evalRes.json();

  if(data.error) {
    indexLocation = (indexLocation + 1) % locations.length;
    return;
  }

  // --- Tentukan status dari FORECAST, bukan dari backend ---
  let forecastStatus = "MINIM POTENSI BANJIR";
  let forecastFloodTime = null;

  if(forecastRes.ok){
    const fData = await forecastRes.json();
    const threshold = !evalData.error && evalData.threshold != null
      ? evalData.threshold
      : 0.12; // fallback default

    const values = fData.values || [];
    const steps  = fData.steps  || [];

    // Hitung berapa langkah yang melewati threshold
    const FLOOD_STEP_THRESHOLD = 4; // minimal langkah untuk dianggap banjir
    let exceedCount = 0;
    let firstExceedStep = null;

    for(let i = 0; i < values.length; i++){
      if(values[i] >= threshold){
        exceedCount++;
        if(firstExceedStep === null) firstExceedStep = steps[i] || null;
      }
    }

    if(exceedCount > FLOOD_STEP_THRESHOLD){
      forecastStatus = "BANJIR";
      forecastFloodTime = firstExceedStep;
    }
  }

  const banner       = document.getElementById("statusBanner");
  const locationName = document.getElementById("locationName");

  locationName.innerText = formatNama(location);

  // Tampilkan status dari forecast
  if(forecastStatus === "BANJIR"){
    banner.innerText          = "⚠ POTENSI BANJIR";
    banner.style.background   = "#f59e0b";
    banner.style.color        = "#fff";
  } else {
    banner.innerText          = "✔ KONDISI MINIM POTENSI BANJIR";
    banner.style.background   = "#16a34a";
    banner.style.color        = "#fff";
  }

  // Tinggi pasang surut dari data aktual
  if(data.elevm != null){
    probValue.innerText = data.elevm.toFixed(2) + " m";
  } else {
    probValue.innerText = "-";
  }

  indexLocation = (indexLocation + 1) % locations.length;

  // --- Pesan peringatan ---
  const namaLokasi = formatNama(location);
  let message = "";

  if(forecastStatus === "BANJIR"){
    message = forecastFloodTime
      ? `⚠️ ${namaLokasi}: forecast menunjukkan potensi banjir pada ${forecastFloodTime}.`
      : `⚠️ ${namaLokasi}: forecast menunjukkan potensi banjir dalam 48 jam ke depan.`;
  } else {
    message = `✅ ${namaLokasi}: forecast 48 jam ke depan dalam kondisi minim potensi banjir.`;
  }

  // Hindari duplikat lokasi yang sama
  alertHistory = alertHistory.filter(item => !item.includes(namaLokasi));
  alertHistory.unshift(message);
  if(alertHistory.length > 4) alertHistory.pop();

  const alertList = document.getElementById("alertList");
  alertList.innerHTML = "";
  alertHistory.forEach(item => {
    const li = document.createElement("li");
    li.textContent = item;
    // Warna teks sesuai status
    li.style.color = item.startsWith("⚠️") ? "#c0392b" : "#27ae60";
    li.style.marginBottom = "4px";
    alertList.appendChild(li);
  });
}


document.addEventListener("DOMContentLoaded",()=>{

  loadLocations();

});
