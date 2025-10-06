/**
 * AI Chop Shop — Coming Soon Landing
 * - GTA V–inspired stacked logo and colorful stripes
 * - Prominent email sign-up
 * - LocalStorage persistence for simple demo capture
 */

const $ = (sel) => document.querySelector(sel);
const $ = (sel) => Array.from(document.querySelectorAll(sel));

document.addEventListener("DOMContentLoaded", () => {
  const form = $("#signup-form");
  const emailInput = $("#email-input");
  const button = $("#signup-button");
  const message = $("#signup-message");
  const year = $("#year");

  if (year) {
    year.textContent = String(new Date().getFullYear());
  }

  const STORAGE_KEY = "ai_chop_shop_emails";
  const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
  const existing = (saved && Array.isArray(saved)) ? saved : [];

  function setStatus(text, kind) {
    message.textContent = text;
    message.classList.remove("success", "error");
    if (kind) message.classList.add(kind);
  }

  function isValidEmail(value) {
    const v = String(value || "").trim();
    // Simple email check for demo purposes
    return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(v);
  }

  form?.addEventListener("submit", (e) => {
    e.preventDefault();
    const email = String(emailInput?.value || "").trim();

    if (!isValidEmail(email)) {
      setStatus("Please enter a valid email address.", "error");
      return;
    }

    if (existing.includes(email.toLowerCase())) {
      setStatus("You're already on the list. We’ll be in touch!", "success");
      button.disabled = true;
      return;
    }

    existing.push(email.toLowerCase());
    localStorage.setItem(STORAGE_KEY, JSON.stringify(existing));
    setStatus("Thanks! You’re on the list. We’ll email you when the doors open.", "success");
    button.disabled = true;
  });
});