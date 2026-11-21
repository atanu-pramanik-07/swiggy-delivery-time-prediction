const PRESETS = {
  indore: {
    ID: "0x4607",
    Delivery_person_ID: "INDORES13DEL02",
    Delivery_person_Age: 34,
    Delivery_person_Ratings: 4.8,
    Restaurant_latitude: 22.745049,
    Restaurant_longitude: 75.892471,
    Delivery_location_latitude: 22.765049,
    Delivery_location_longitude: 75.912471,
    Order_Date: "2022-03-19",
    Time_Orderd: "11:30:00",
    Time_Order_picked: "11:40:00",
    Weatherconditions: "conditions Sunny",
    Road_traffic_density: "Low",
    Vehicle_condition: 2,
    Type_of_order: "Snack",
    Type_of_vehicle: "motorcycle",
    multiple_deliveries: 0,
    Festival: "No",
    City: "Urban",
  },
  bengaluru: {
    ID: "0xb379",
    Delivery_person_ID: "BANGRES18DEL02",
    Delivery_person_Age: 28,
    Delivery_person_Ratings: 4.4,
    Restaurant_latitude: 12.913041,
    Restaurant_longitude: 77.683237,
    Delivery_location_latitude: 13.043041,
    Delivery_location_longitude: 77.813237,
    Order_Date: "2022-03-25",
    Time_Orderd: "19:45:00",
    Time_Order_picked: "20:00:00",
    Weatherconditions: "conditions Cloudy",
    Road_traffic_density: "Jam",
    Vehicle_condition: 1,
    Type_of_order: "Meal",
    Type_of_vehicle: "scooter",
    multiple_deliveries: 2,
    Festival: "No",
    City: "Metropolitian",
  },
  hyderabad: {
    ID: "0x95b4",
    Delivery_person_ID: "HYDRES09DEL03",
    Delivery_person_Age: 36,
    Delivery_person_Ratings: 4.2,
    Restaurant_latitude: 17.431477,
    Restaurant_longitude: 78.40035,
    Delivery_location_latitude: 17.501477,
    Delivery_location_longitude: 78.47035,
    Order_Date: "2022-04-02",
    Time_Orderd: "21:15:00",
    Time_Order_picked: "21:30:00",
    Weatherconditions: "conditions Stormy",
    Road_traffic_density: "High",
    Vehicle_condition: 0,
    Type_of_order: "Buffet",
    Type_of_vehicle: "motorcycle",
    multiple_deliveries: 1,
    Festival: "Yes",
    City: "Metropolitian",
  },
};

const form = document.getElementById("predictForm");
const submitBtn = document.getElementById("submitBtn");
const etaCard = document.getElementById("etaCard");
const etaNumber = document.getElementById("etaNumber");
const etaRange = document.getElementById("etaRange");
const paceBadge = document.getElementById("paceBadge");
const routeVisual = document.getElementById("routeVisual");
const errorBanner = document.getElementById("errorBanner");
const apiStatus = document.getElementById("apiStatus");
const payloadPreview = document.getElementById("payloadPreview");
const historyList = document.getElementById("historyList");
const clearHistoryBtn = document.getElementById("clearHistoryBtn");

const ageInput = document.getElementById("Delivery_person_Age");
const ratingInput = document.getElementById("Delivery_person_Ratings");
const ageValue = document.getElementById("ageValue");
const ratingValue = document.getElementById("ratingValue");

const inlineDistanceChip = document.getElementById("inlineDistanceChip");
const metricDistance = document.getElementById("metricDistance");
const metricDistType = document.getElementById("metricDistType");
const metricPickup = document.getElementById("metricPickup");
const metricTimeOfDay = document.getElementById("metricTimeOfDay");
const metricDayType = document.getElementById("metricDayType");
const metricCityCode = document.getElementById("metricCityCode");

let lastDisplayedEta = 0;
let predictionHistory = [];

/** Format YYYY-MM-DD -> DD-MM-YYYY for FastAPI schema */
function formatDateForApi(isoDate) {
  if (!isoDate || !isoDate.includes("-")) return isoDate;
  const [yyyy, mm, dd] = isoDate.split("-");
  return `${dd}-${mm}-${yyyy}`;
}

/** Ensure HH:MM:SS 24-hour format even when browser drops :00 seconds */
function normalizeTime(timeStr) {
  if (!timeStr) return "00:00:00";
  const parts = timeStr.split(":");
  const hh = (parts[0] || "00").padStart(2, "0");
  const mm = (parts[1] || "00").padStart(2, "0");
  const ss = (parts[2] || "00").padStart(2, "0");
  return `${hh}:${mm}:${ss}`;
}

/** Haversine distance in km (matches backend data_cleaning.py) */
function haversineKm(lat1, lon1, lat2, lon2) {
  const toRad = (deg) => (deg * Math.PI) / 180;
  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) ** 2;
  return 6371 * 2 * Math.asin(Math.sqrt(a));
}

function getDistanceBucket(km) {
  if (Number.isNaN(km)) return "—";
  if (km < 5) return "short";
  if (km < 10) return "medium";
  if (km < 15) return "long";
  if (km < 25) return "very_long";
  return "extreme";
}

function getTimeOfDay(hour) {
  if (hour <= 6) return "after_midnight";
  if (hour <= 12) return "morning";
  if (hour <= 17) return "afternoon";
  if (hour <= 20) return "evening";
  return "night";
}

function getPickupMinutes(orderTime, pickedTime) {
  const toSeconds = (t) => {
    const [h, m, s] = normalizeTime(t).split(":").map(Number);
    return h * 3600 + m * 60 + s;
  };
  let diff = toSeconds(pickedTime) - toSeconds(orderTime);
  if (diff < 0) diff += 86400; // Midnight crossing
  return Math.round(diff / 60);
}

/** Build exact JSON payload matching FastAPI Pydantic Data model */
function buildPayload() {
  return {
    ID: document.getElementById("ID").value.trim(),
    Delivery_person_ID: document.getElementById("Delivery_person_ID").value.trim().toUpperCase(),
    Delivery_person_Age: parseInt(ageInput.value, 10),
    Delivery_person_Ratings: parseFloat(ratingInput.value),
    Restaurant_latitude: parseFloat(document.getElementById("Restaurant_latitude").value),
    Restaurant_longitude: parseFloat(document.getElementById("Restaurant_longitude").value),
    Delivery_location_latitude: parseFloat(document.getElementById("Delivery_location_latitude").value),
    Delivery_location_longitude: parseFloat(document.getElementById("Delivery_location_longitude").value),
    Order_Date: formatDateForApi(document.getElementById("Order_Date").value),
    Time_Orderd: normalizeTime(document.getElementById("Time_Orderd").value),
    Time_Order_picked: normalizeTime(document.getElementById("Time_Order_picked").value),
    Weatherconditions: document.getElementById("Weatherconditions").value,
    Road_traffic_density: document.getElementById("Road_traffic_density").value,
    Vehicle_condition: parseInt(document.getElementById("Vehicle_condition").value, 10),
    Type_of_order: document.getElementById("Type_of_order").value,
    Type_of_vehicle: document.getElementById("Type_of_vehicle").value,
    multiple_deliveries: parseInt(document.getElementById("multiple_deliveries").value, 10),
    Festival: document.getElementById("Festival").value,
    City: document.getElementById("City").value,
  };
}

/** Update live derived metrics & JSON preview */
function updateLiveTelemetry() {
  ageValue.textContent = `${ageInput.value} yrs`;
  ratingValue.textContent = `${parseFloat(ratingInput.value).toFixed(1)} ★`;

  const payload = buildPayload();
  const km = haversineKm(
    payload.Restaurant_latitude,
    payload.Restaurant_longitude,
    payload.Delivery_location_latitude,
    payload.Delivery_location_longitude
  );

  const kmText = Number.isFinite(km) ? `${km.toFixed(2)} km` : "—";
  inlineDistanceChip.textContent = kmText;
  metricDistance.textContent = kmText;
  metricDistType.textContent = getDistanceBucket(km);

  const pickupMin = getPickupMinutes(payload.Time_Orderd, payload.Time_Order_picked);
  metricPickup.textContent = Number.isFinite(pickupMin) ? `${pickupMin} min` : "—";

  const hour = parseInt(payload.Time_Orderd.split(":")[0], 10);
  metricTimeOfDay.textContent = Number.isFinite(hour) ? getTimeOfDay(hour) : "—";

  const isoDate = document.getElementById("Order_Date").value;
  if (isoDate) {
    const dayIdx = new Date(`${isoDate}T00:00:00`).getDay();
    metricDayType.textContent = dayIdx === 0 || dayIdx === 6 ? "weekend" : "weekday";
  }

  const cityCode = payload.Delivery_person_ID.split("RES")[0] || "—";
  metricCityCode.textContent = cityCode;

  payloadPreview.textContent = JSON.stringify(payload, null, 2);
}

/** Smooth count-up animation for ETA number */
function animateEtaValue(targetVal) {
  const startVal = lastDisplayedEta || 0;
  const duration = 620;
  const startTime = performance.now();

  function step(now) {
    const progress = Math.min((now - startTime) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 4);
    const current = startVal + (targetVal - startVal) * eased;
    etaNumber.textContent = current.toFixed(1);
    if (progress < 1) {
      requestAnimationFrame(step);
    } else {
      lastDisplayedEta = targetVal;
    }
  }
  requestAnimationFrame(step);
}

function setStatus(state, text) {
  apiStatus.className = `status-pill ${state}`;
  apiStatus.querySelector(".status-text").textContent = text;
}

function updatePaceBadge(minutes) {
  paceBadge.className = "pace-badge";
  if (minutes < 22) {
    paceBadge.classList.add("fast");
    paceBadge.textContent = "Fast Dispatch";
  } else if (minutes < 34) {
    paceBadge.classList.add("moderate");
    paceBadge.textContent = "Standard ETA";
  } else {
    paceBadge.classList.add("slow");
    paceBadge.textContent = "High Delay Risk";
  }
}

function showError(message) {
  errorBanner.textContent = message;
  errorBanner.hidden = false;
}

function hideError() {
  errorBanner.hidden = true;
  errorBanner.textContent = "";
}

/** Render session prediction history */
function renderHistory() {
  if (predictionHistory.length === 0) {
    historyList.innerHTML = `<li class="history-empty">No predictions yet in this session.</li>`;
    clearHistoryBtn.hidden = true;
    return;
  }

  clearHistoryBtn.hidden = false;
  historyList.innerHTML = predictionHistory
    .map(
      (item, idx) => `
      <li class="history-item" data-index="${idx}" title="Click to restore parameters">
        <div class="history-meta">
          <span class="history-title">${item.cityCode} · ${item.traffic} Traffic</span>
          <span class="history-sub mono">${item.distance} · ${item.weather}</span>
        </div>
        <span class="history-eta mono">${item.eta.toFixed(1)} min</span>
      </li>
    `
    )
    .join("");
}

/** Set segmented control programmatically */
function setSegmentedValue(fieldName, value) {
  const hiddenInput = document.getElementById(fieldName);
  const group = document.querySelector(`.segmented[data-field="${fieldName}"]`);
  if (!hiddenInput || !group) return;

  hiddenInput.value = String(value);
  group.querySelectorAll(".seg-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.value === String(value));
  });
}

/** Apply a preset scenario */
function applyPreset(key) {
  const preset = PRESETS[key];
  if (!preset) return;

  document.querySelectorAll(".preset-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.preset === key);
  });

  Object.entries(preset).forEach(([field, value]) => {
    const segGroup = document.querySelector(`.segmented[data-field="${field}"]`);
    if (segGroup) {
      setSegmentedValue(field, value);
    } else {
      const el = document.getElementById(field);
      if (el) el.value = value;
    }
  });

  hideError();
  updateLiveTelemetry();
}

/** Segmented button click handlers */
document.querySelectorAll(".segmented").forEach((group) => {
  const fieldName = group.dataset.field;
  group.addEventListener("click", (e) => {
    const btn = e.target.closest(".seg-btn");
    if (!btn) return;
    setSegmentedValue(fieldName, btn.dataset.value);
    updateLiveTelemetry();
  });
});

/** Preset click handlers */
document.querySelectorAll(".preset-btn").forEach((btn) => {
  btn.addEventListener("click", () => applyPreset(btn.dataset.preset));
});

/** Live form input listeners */
form.addEventListener("input", () => {
  document.querySelectorAll(".preset-btn.active").forEach((b) => b.classList.remove("active"));
  updateLiveTelemetry();
});
form.addEventListener("change", updateLiveTelemetry);

/** Restore from history click */
historyList.addEventListener("click", (e) => {
  const item = e.target.closest(".history-item");
  if (!item) return;
  const entry = predictionHistory[Number(item.dataset.index)];
  if (!entry) return;

  Object.entries(entry.formState).forEach(([field, value]) => {
    const segGroup = document.querySelector(`.segmented[data-field="${field}"]`);
    if (segGroup) {
      setSegmentedValue(field, value);
    } else {
      const el = document.getElementById(field);
      if (el) el.value = value;
    }
  });
  updateLiveTelemetry();
});

clearHistoryBtn.addEventListener("click", () => {
  predictionHistory = [];
  renderHistory();
});

/** Submit prediction request */
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  hideError();

  // Basic HTML5 validity check
  if (!form.checkValidity()) {
    const firstInvalid = form.querySelector(":invalid");
    if (firstInvalid) {
      firstInvalid.classList.add("invalid");
      firstInvalid.focus();
      setTimeout(() => firstInvalid.classList.remove("invalid"), 2200);
    }
    showError("Please check highlighted fields — Rider ID must follow format like INDORES13DEL02 and coordinates must be within India.");
    return;
  }

  const payload = buildPayload();
  const formSnapshot = {
    ...payload,
    Order_Date: document.getElementById("Order_Date").value,
  };

  submitBtn.disabled = true;
  submitBtn.classList.add("loading");
  etaCard.classList.add("loading");
  routeVisual.classList.remove("done");
  routeVisual.classList.add("running");
  setStatus("loading", "Predicting...");

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const body = await response.json();

    if (!response.ok) {
      if (Array.isArray(body.detail)) {
        const msgs = body.detail
          .map((err) => `${err.loc[err.loc.length - 1]}: ${err.msg}`)
          .join(" | ");
        throw new Error(msgs);
      }
      throw new Error(body.detail || `Request failed with status ${response.status}`);
    }

    const eta = Number(body.pred);
    animateEtaValue(eta);
    updatePaceBadge(eta);

    const low = Math.max(1, Math.round(eta - 3));
    const high = Math.round(eta + 3);
    etaRange.textContent = `Estimated arrival window: ${low} – ${high} min (${metricDistance.textContent} · ${payload.Road_traffic_density} traffic)`;

    routeVisual.classList.remove("running");
    routeVisual.classList.add("done");
    setStatus("", "Ready");

    // Push to session history
    predictionHistory.unshift({
      cityCode: metricCityCode.textContent,
      traffic: payload.Road_traffic_density,
      distance: metricDistance.textContent,
      weather: payload.Weatherconditions.replace("conditions ", ""),
      eta,
      formState: formSnapshot,
    });
    predictionHistory = predictionHistory.slice(0, 5);
    renderHistory();
  } catch (err) {
    routeVisual.classList.remove("running");
    setStatus("error", "Error");
    showError(err.message || "Unable to reach /predict endpoint.");
  } finally {
    submitBtn.disabled = false;
    submitBtn.classList.remove("loading");
    etaCard.classList.remove("loading");
  }
});

// Initialize on load
updateLiveTelemetry();