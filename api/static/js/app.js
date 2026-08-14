"use strict";

const EXAMPLE = {
  tenure: 5, MonthlyCharges: 89.9, TotalCharges: 450.5,
  gender: "Female", SeniorCitizen: 0, Partner: "No", Dependents: "No",
  PhoneService: "Yes", MultipleLines: "No", InternetService: "Fiber optic",
  OnlineSecurity: "No", OnlineBackup: "Yes", DeviceProtection: "No",
  TechSupport: "No", StreamingTV: "Yes", StreamingMovies: "Yes",
  Contract: "Month-to-month", PaperlessBilling: "Yes", PaymentMethod: "Electronic check"
};

const form = document.querySelector("#churn-form");
const submitButton = document.querySelector("#predict-button");
const formError = document.querySelector("#form-error");
const phoneService = document.querySelector("#phone-service");
const multipleLines = document.querySelector("#multiple-lines");
const internetService = document.querySelector("#internet-service");
const internetAddons = [...document.querySelectorAll(".internet-addon")];

function syncPhoneFields() {
  const noPhone = phoneService.value === "No";
  multipleLines.value = noPhone ? "No phone service" : (multipleLines.value === "No phone service" ? "No" : multipleLines.value);
  multipleLines.disabled = noPhone;
}

function syncInternetFields() {
  const noInternet = internetService.value === "No";
  internetAddons.forEach((field) => {
    field.value = noInternet ? "No internet service" : (field.value === "No internet service" ? "No" : field.value);
    field.disabled = noInternet;
  });
}

function fillForm(values) {
  Object.entries(values).forEach(([name, value]) => {
    const field = form.elements.namedItem(name);
    if (field) field.value = String(value);
  });
  syncPhoneFields(); syncInternetFields();
}

function payloadFromForm() {
  const data = Object.fromEntries(new FormData(form).entries());
  data.tenure = Number(data.tenure);
  data.MonthlyCharges = Number(data.MonthlyCharges);
  data.TotalCharges = Number(data.TotalCharges);
  data.SeniorCitizen = Number(data.SeniorCitizen);
  if (phoneService.value === "No") data.MultipleLines = "No phone service";
  if (internetService.value === "No") internetAddons.forEach((field) => { data[field.name] = "No internet service"; });
  return data;
}

function riskPresentation(probability) {
  if (probability < 0.30) return { label: "Low", className: "risk-low", color: "#18794e" };
  if (probability < 0.60) return { label: "Moderate", className: "risk-moderate", color: "#b66b09" };
  return { label: "High", className: "risk-high", color: "#d14343" };
}

function showResult(result) {
  const percentage = result.churn_probability * 100;
  const risk = riskPresentation(result.churn_probability);
  document.querySelector("#result-empty").classList.add("hidden");
  document.querySelector("#result-content").classList.remove("hidden");
  document.querySelector("#probability-value").textContent = `${percentage.toFixed(1)}%`;
  const ring = document.querySelector("#probability-ring");
  ring.style.setProperty("--probability", `${percentage * 3.6}deg`);
  ring.style.setProperty("--ring-color", risk.color);
  ring.setAttribute("aria-label", `Churn probability ${percentage.toFixed(1)} percent`);
  document.querySelector("#prediction-label").textContent = result.churn_label;
  document.querySelector("#risk-level").textContent = `${risk.label} (presentation)`;
  document.querySelector("#threshold-value").textContent = `${(result.threshold * 100).toFixed(0)}%`;
  document.querySelector("#prediction-code").textContent = String(result.churn_prediction);
  const badge = document.querySelector("#risk-badge");
  badge.textContent = `${risk.label} risk`;
  badge.className = `risk-badge ${risk.className}`;
  document.querySelector("#result-panel").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function resetExperience() {
  form.reset(); syncPhoneFields(); syncInternetFields();
  formError.classList.add("hidden"); formError.textContent = "";
  document.querySelector("#result-content").classList.add("hidden");
  document.querySelector("#result-empty").classList.remove("hidden");
}

async function submitPrediction(event) {
  event.preventDefault();
  formError.classList.add("hidden");
  if (!form.reportValidity()) return;
  submitButton.disabled = true;
  submitButton.querySelector("span").textContent = "Predicting…";
  try {
    const response = await fetch("/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payloadFromForm()) });
    if (!response.ok) {
      if (response.status === 422) throw new Error("Some customer information is invalid or inconsistent. Please review the highlighted choices.");
      throw new Error("The prediction service could not complete this request. Please try again.");
    }
    showResult(await response.json());
  } catch (error) {
    formError.textContent = error instanceof TypeError ? "The prediction service is unavailable. Check your connection and try again." : error.message;
    formError.classList.remove("hidden");
    formError.scrollIntoView({ behavior: "smooth", block: "center" });
  } finally {
    submitButton.disabled = false;
    submitButton.querySelector("span").textContent = "Predict Churn Risk";
  }
}

async function checkHealth() {
  const status = document.querySelector("#service-status");
  try {
    const response = await fetch("/health");
    if (!response.ok || !(await response.json()).model_loaded) throw new Error();
    status.className = "service-status ready"; status.lastElementChild.textContent = "Model ready";
  } catch {
    status.className = "service-status unavailable"; status.lastElementChild.textContent = "Service unavailable";
  }
}

phoneService.addEventListener("change", syncPhoneFields);
internetService.addEventListener("change", syncInternetFields);
form.addEventListener("submit", submitPrediction);
document.querySelector("#load-example").addEventListener("click", () => fillForm(EXAMPLE));
document.querySelector("#reset-form").addEventListener("click", resetExperience);
document.querySelector("#new-prediction").addEventListener("click", () => { resetExperience(); document.querySelector("#form-title").scrollIntoView({ behavior: "smooth" }); });
syncPhoneFields(); syncInternetFields(); checkHealth();
