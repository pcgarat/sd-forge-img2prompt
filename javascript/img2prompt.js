/* Dual-thumb word-range sliders for Image → Prompt */
(function () {
  function bridgeInput(bridgeId) {
    const root = document.getElementById(bridgeId);
    if (!root) return null;
    return root.querySelector("textarea, input");
  }

  function setNativeValue(el, value) {
    // Gradio/React ignores plain el.value=…; use the native setter.
    const proto =
      el.tagName === "TEXTAREA"
        ? window.HTMLTextAreaElement.prototype
        : window.HTMLInputElement.prototype;
    const desc = Object.getOwnPropertyDescriptor(proto, "value");
    if (desc && desc.set) {
      desc.set.call(el, value);
    } else {
      el.value = value;
    }
  }

  function setBridge(bridgeId, lo, hi) {
    const input = bridgeInput(bridgeId);
    if (!input) return false;
    const next = String(lo) + "," + String(hi);
    if (input.value !== next) {
      setNativeValue(input, next);
    }
    // Forge WebUI: plain DOM events are not enough; updateInput pushes the
    // value into Gradio's internal state so Python receives it on click.
    // Call even when DOM already matches — Gradio state can still be stale.
    if (typeof updateInput === "function") {
      updateInput(input);
    } else {
      input.dispatchEvent(new Event("input", { bubbles: true }));
      input.dispatchEvent(new Event("change", { bubbles: true }));
    }
    return true;
  }

  function paint(widget) {
    const min = Number(widget.dataset.min);
    const max = Number(widget.dataset.max);
    const lo = Number(widget.dataset.lo);
    const hi = Number(widget.dataset.hi);
    const range = Math.max(max - min, 1);
    const loPct = ((lo - min) / range) * 100;
    const hiPct = ((hi - min) / range) * 100;
    widget.style.setProperty("--img2prompt-lo", loPct + "%");
    widget.style.setProperty("--img2prompt-hi", hiPct + "%");
    const loVal = widget.querySelector(".lo-val");
    const hiVal = widget.querySelector(".hi-val");
    if (loVal) loVal.textContent = String(lo);
    if (hiVal) hiVal.textContent = String(hi);
  }

  function readWidgetRange(widget) {
    const loEl = widget.querySelector(".img2prompt-dual-lo");
    const hiEl = widget.querySelector(".img2prompt-dual-hi");
    if (!loEl || !hiEl) return null;
    let lo = Number(loEl.value);
    let hi = Number(hiEl.value);
    if (Number.isNaN(lo) || Number.isNaN(hi)) return null;
    if (lo > hi) {
      const t = lo;
      lo = hi;
      hi = t;
      loEl.value = String(lo);
      hiEl.value = String(hi);
    }
    widget.dataset.lo = String(lo);
    widget.dataset.hi = String(hi);
    return { lo: lo, hi: hi };
  }

  function syncWidget(widget) {
    const bridgeId = widget.dataset.bridge;
    const range = readWidgetRange(widget);
    if (!range || !bridgeId) return;
    paint(widget);
    setBridge(bridgeId, range.lo, range.hi);
  }

  function bindWidget(widget) {
    if (widget.dataset.bound === "1") return;
    widget.dataset.bound = "1";
    const loEl = widget.querySelector(".img2prompt-dual-lo");
    const hiEl = widget.querySelector(".img2prompt-dual-hi");
    if (!loEl || !hiEl) return;

    const syncFromInputs = () => syncWidget(widget);

    loEl.addEventListener("input", syncFromInputs);
    hiEl.addEventListener("input", syncFromInputs);
    loEl.addEventListener("change", syncFromInputs);
    hiEl.addEventListener("change", syncFromInputs);
    syncFromInputs();
  }

  function initRanges() {
    document.querySelectorAll(".img2prompt-dual").forEach(bindWidget);
  }

  /** Force every dual-range into its Gradio bridge textbox (call before Generate/Detail). */
  function syncRanges() {
    document.querySelectorAll(".img2prompt-dual").forEach(syncWidget);
  }

  window.img2promptInitRanges = initRanges;
  window.img2promptSyncRanges = syncRanges;

  if (typeof onUiLoaded === "function") {
    onUiLoaded(initRanges);
  } else {
    document.addEventListener("DOMContentLoaded", initRanges);
  }
  // Gradio re-renders tabs; keep widgets alive
  setInterval(initRanges, 1500);
})();
