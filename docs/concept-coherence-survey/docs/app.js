"use strict";
(() => {
  const $ = id => document.getElementById(id);
  const base = (window.SURVEY_CONFIG?.apiBase || "").replace(/\/$/, "");
  let token = sessionStorage.getItem("survey-token") || "";
  let study, sample, imageURLs = [], busy = false, loading = false, submitting = false, changed = false, leaseLost = false;
  let saveTimer, heartbeat, lastClock = performance.now(), clockRunning = false, tickQueue = Promise.resolve();
  let saveQueue = Promise.resolve(), viewerURL, viewerSequence = 0;
  const screens = ["login", "instructions", "workspace", "complete"];
  const announce = (text = "") => { $("notice").textContent = text; $("notice").hidden = !text; };
  function screen(id) { screens.forEach(x => $(x).hidden = x !== id); window.scrollTo({top: 0}); }
  function savedStatus(text) { $("save-status").textContent = text; }
  function setAccount() { $("account-label").textContent = study?.email || ""; $("sign-out").hidden = !token; }
  function localKey() { return `survey-draft:${study.participant_id}:${sample.sample_alias}`; }
  async function api(path, options = {}) {
    let response;
    try {
      response = await fetch(base + path, {...options, headers: {
        ...(token ? {Authorization: `Bearer ${token}`} : {}),
        ...(options.body ? {"Content-Type": "application/json"} : {}), ...(options.headers || {})
      }});
    } catch (_) { throw new Error("Cannot reach the study server. Your current draft stays in this tab. Check your connection and try again."); }
    if (!response.ok) {
      let detail = "The request could not be completed.";
      try { const data = await response.json(); detail = typeof data.detail === "string" ? data.detail : "Check that all responses are valid."; } catch (_) {}
      const error = new Error(detail); error.status = response.status; throw error;
    }
    return options.blob ? response.blob() : response.json();
  }
  const post = (path, body) => api(path, {method: "POST", ...(body ? {body: JSON.stringify(body)} : {})});
  function withBusy(form, action) {
    if (busy) return;
    busy = true; const buttons = [...form.querySelectorAll("button")]; buttons.forEach(b => b.disabled = true);
    announce();
    return action().catch(e => announce(e.message)).finally(() => {busy = false; buttons.forEach(b => b.disabled = false);});
  }
  $("email-form").onsubmit = e => {
    e.preventDefault();
    withBusy(e.currentTarget, async () => {
      await post("/api/auth/request-code", {email: $("email").value.trim()});
      $("email-form").hidden = true; $("code-form").hidden = false; $("code").focus();
      announce("If your email is invited, a code has been sent. It expires in 10 minutes. Check your spam folder too.");
    });
  };
  $("change-email").onclick = () => { $("code-form").hidden = true; $("email-form").hidden = false; $("code").value = ""; announce(); };
  $("code-form").onsubmit = e => {
    e.preventDefault(); withBusy(e.currentTarget, async () => {
      const result = await post("/api/auth/verify", {email: $("email").value.trim(), code: $("code").value.trim()});
      token = result.token; sessionStorage.setItem("survey-token", token); study = result; setAccount(); prepare();
    });
  };
  function prepare() {
    announce(); setAccount();
    if (study.complete) { finish(); return; }
    $("study-size").textContent = `This study contains ${study.total_samples} groups. You have submitted ${study.completed_samples}.`;
    $("context-instructions").textContent = study.frame_context ? "You may open a boxed full frame to locate a patch. The red rectangle marks its location. Full-frame views are recorded; judge the pattern inside the patch." : "This study uses patches only. Full-frame context is disabled.";
    $("begin").textContent = study.completed_samples ? "Resume annotation →" : "Begin annotation →";
    screen("instructions");
  }
  function answer() {
    return {description: $("description").value,
      coherence: Number(document.querySelector('input[name="coherence"]:checked')?.value) || null,
      outlier_mode: document.querySelector('input[name="outlier-mode"]:checked')?.value || null,
      outliers: [...document.querySelectorAll('#outlier-options input:checked')].map(x => x.value)};
  }
  function restore(a) {
    $("annotation-form").reset(); $("description").value = a.description || "";
    if (a.coherence) document.querySelector(`input[name="coherence"][value="${a.coherence}"]`)?.click();
    if (["selected","none","cannot_determine"].includes(a.outlier_mode)) document.querySelector(`input[name="outlier-mode"][value="${a.outlier_mode}"]`)?.click();
    [...$("outlier-options").querySelectorAll("input")].forEach(x => x.checked = (a.outliers || []).includes(x.value));
    $("ratings").hidden = !a.coherence && !a.outlier_mode; $("show-ratings").hidden = !$("ratings").hidden;
    syncOutliers();
  }
  function syncOutliers() {
    const selected = document.querySelector('input[name="outlier-mode"]:checked')?.value === "selected";
    $("outlier-options").querySelectorAll("input").forEach(x => {x.disabled = !selected; if (!selected) x.checked = false;});
  }
  function dirty() {
    if (loading || submitting || !sample) return;
    changed = true; sessionStorage.setItem(localKey(), JSON.stringify(answer())); savedStatus("Unsaved changes");
    clearTimeout(saveTimer); saveTimer = setTimeout(() => saveDraft().catch(e => {savedStatus("Draft kept in this tab"); announce(e.message);}), 650);
  }
  function saveDraft() {
    if (!changed || !sample) return saveQueue;
    const snapshot = answer(), alias = sample.sample_alias, visit = sample.visit_id, key = localKey();
    // Serialize writes so an older autosave cannot overwrite a newer draft.
    saveQueue = saveQueue.catch(() => {}).then(async () => {
      await api(`/api/samples/${alias}/draft`, {method: "PUT", body: JSON.stringify(snapshot), headers: {"X-Visit-ID": visit}});
      if (sample?.sample_alias === alias && JSON.stringify(answer()) === JSON.stringify(snapshot)) {
        changed = false; sessionStorage.removeItem(key); savedStatus("Draft saved");
      }
    });
    return saveQueue;
  }
  function queueTick(keepalive = false) {
    if (!sample || !clockRunning) return tickQueue;
    const now = performance.now();
    const seconds = Math.max(0, Math.min((now - lastClock) / 1000, 15)); lastClock = now;
    const snapshot = sample;
    tickQueue = tickQueue.catch(() => {}).then(() => api(`/api/samples/${snapshot.sample_alias}/tick`, {
      method: "POST", body: JSON.stringify({visit_id: snapshot.visit_id, seconds}), keepalive
    })).catch(e => {
      if (e.status === 409) { leaseLost = true; clockRunning = false; $("submit-answer").disabled = true; }
      savedStatus("Timing sync interrupted"); announce(e.message); throw e;
    });
    return tickQueue;
  }
  function stopClock() { clearInterval(heartbeat); clockRunning = false; }
  function beginClock() {
    clearInterval(heartbeat); clockRunning = !document.hidden && document.hasFocus(); lastClock = performance.now();
    heartbeat = setInterval(() => { if (clockRunning) queueTick().catch(() => {}); }, 5000);
  }
  async function loadGroup() {
    if (loading) return;
    loading = true; changed = false; leaseLost = false; stopClock(); clearTimeout(saveTimer); announce();
    $("submit-answer").disabled = true;
    try {
      study = await api("/api/study");
      if (study.complete) { finish(); return; }
      const alias = `S${String(study.completed_samples + 1).padStart(3,"0")}`;
      sample = await post(`/api/samples/${alias}/start`);
      screen("workspace");
      $("sample-title").textContent = `Group ${study.completed_samples + 1}`;
      $("progress-label").textContent = `${study.completed_samples} / ${study.total_samples} completed`;
      $("progress").max = study.total_samples; $("progress").value = study.completed_samples;
      $("patch-count").textContent = `${sample.patches.length} patches`;
      $("submit-answer").textContent = study.completed_samples + 1 === study.total_samples ? "Submit final group and finish ✓" : "Submit and continue →";
      imageURLs.forEach(URL.revokeObjectURL); imageURLs = []; $("patch-grid").replaceChildren(); $("outlier-options").replaceChildren();
      const pending = sample.patches.map(async id => {
        const card = document.createElement("article"); card.className = "patch-card";
        const button = document.createElement("button"); button.type = "button"; button.className = "patch-image-button";
        button.setAttribute("aria-label", `Enlarge patch ${id}`);
        const img = document.createElement("img"); img.alt = `Patch ${id}`; button.append(img);
        const bottom = document.createElement("div"); bottom.className = "patch-bottom";
        const label = document.createElement("span"); label.className = "patch-label"; label.textContent = id; bottom.append(label);
        if (study.frame_context) {
          const frame = document.createElement("button"); frame.type = "button"; frame.className = "text-button frame-button"; frame.textContent = "View full frame";
          frame.onclick = () => viewImage(id, "frame"); bottom.append(frame);
        }
        card.append(button,bottom); $("patch-grid").append(card);
        const outlier = document.createElement("label"), checkbox = document.createElement("input");
        checkbox.type = "checkbox"; checkbox.value = id; outlier.append(checkbox,document.createTextNode(id)); $("outlier-options").append(outlier);
        const blob = await api(`/api/images/${alias}/${id}/patch`, {blob: true});
        const url = URL.createObjectURL(blob); imageURLs.push(url);
        await new Promise((resolve,reject) => {img.onload = resolve; img.onerror = () => reject(new Error("An image could not be decoded. Retry loading the group.")); img.src = url;});
        button.onclick = () => viewImage(id, "patch", url);
      });
      const local = sessionStorage.getItem(localKey());
      let draft = sample.draft;
      if (local) {try {draft = JSON.parse(local);} catch (_) {sessionStorage.removeItem(localKey());}}
      restore(draft); changed = Boolean(local);
      $("reload-images").hidden = true; savedStatus("Loading patches…");
      await Promise.all(pending);
      $("submit-answer").disabled = false; beginClock(); savedStatus(changed ? "Recovered draft" : "Ready");
    } catch (e) { announce(e.message); $("reload-images").hidden = false; }
    finally { loading = false; }
  }
  $("begin").onclick = () => loadGroup(); $("reload-images").onclick = () => loadGroup();
  $("annotation-form").addEventListener("input", dirty);
  $("annotation-form").addEventListener("change", e => {if (e.target.name === "outlier-mode") syncOutliers(); dirty();});
  $("no-pattern").onclick = () => { $("description").value = "No clear shared pattern"; dirty(); };
  $("show-ratings").onclick = () => {
    if (!$("description").value.trim()) { $("description").reportValidity(); $("description").focus(); return; }
    $("ratings").hidden = false; $("show-ratings").hidden = true; document.querySelector('input[name="coherence"]').focus();
  };
  $("annotation-form").onsubmit = async e => {
    e.preventDefault(); if (submitting || loading) return;
    const a = answer();
    if (!a.description.trim() || !a.coherence || !a.outlier_mode || (a.outlier_mode === "selected" && !a.outliers.length)) {
      announce("Please answer all three questions. Select at least one outlier, or choose None / Cannot determine."); return;
    }
    submitting = true; $("annotation-form").inert = true; $("submit-answer").disabled = true; clearTimeout(saveTimer); announce(); savedStatus("Submitting…");
    try {
      // A previous response may have been lost after the server committed it.
      const latest = await api("/api/study");
      if (latest.completed_samples > Number(sample.sample_alias.slice(1)) - 1) {
        sessionStorage.removeItem(localKey()); changed = false; study = latest; stopClock();
        if (study.complete) finish(); else await loadGroup();
        return;
      }
      await saveDraft(); await queueTick(); stopClock();
      const result = await api(`/api/samples/${sample.sample_alias}/submit`, {method: "POST", body: JSON.stringify(a), headers: {"X-Visit-ID": sample.visit_id}});
      sessionStorage.removeItem(localKey()); changed = false; study = result;
      if (study.complete) finish(); else await loadGroup();
    } catch (error) {
      if (error.status === 409) leaseLost = true;
      announce(error.message + " You can retry; duplicate submissions do not create duplicate records.");
      if (!leaseLost) beginClock();
    }
    finally {submitting = false; $("annotation-form").inert = false; $("submit-answer").disabled = leaseLost;}
  };
  async function viewImage(id, kind, existingURL) {
    const sequence = ++viewerSequence;
    if (viewerURL) {URL.revokeObjectURL(viewerURL); viewerURL = null;}
    $("image-title").textContent = kind === "frame" ? `${id} · Full frame` : `${id} · Patch`;
    $("viewer-caption").textContent = kind === "frame" ? "The red rectangle marks the patch. Judge the visual pattern inside that region." : "Enlarged for inspection; no image enhancement is applied.";
    $("viewer-image").hidden = true; $("viewer-status").textContent = "Loading image…";
    if (!$("image-dialog").open) $("image-dialog").showModal();
    try {
      let url = existingURL;
      if (!url) {
        const blob = await api(`/api/images/${sample.sample_alias}/${id}/${kind}`, {blob:true});
        if (sequence !== viewerSequence) return;
        viewerURL = URL.createObjectURL(blob); url = viewerURL;
      }
      if (sequence !== viewerSequence) return;
      $("viewer-image").alt = kind === "frame" ? `Full frame with patch ${id} boxed in red` : `Enlarged patch ${id}`;
      $("viewer-image").src = url; $("viewer-image").hidden = false; $("viewer-status").textContent = "";
    } catch (error) { $("viewer-status").textContent = error.message; }
  }
  $("close-dialog").onclick = () => $("image-dialog").close();
  $("image-dialog").addEventListener("close", () => { viewerSequence++; if (viewerURL) URL.revokeObjectURL(viewerURL); viewerURL = null; });
  function finish() {
    stopClock(); clearTimeout(saveTimer); screen("complete"); announce();
    $("done-count").textContent = study.completed_samples;
    $("done-time").textContent = `${Math.round(study.metadata.total_annotation_time_seconds / 60)} min`;
    $("participant-ref").textContent = `Participant reference: ${study.participant_id}`; sample = null;
  }
  async function download(name) {
    try {
      const blob = await api(`/api/export/${name}`, {blob:true}); const url = URL.createObjectURL(blob);
      const link = document.createElement("a"); link.href = url; link.download = name; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch(e) {announce(e.message);}
  }
  $("download-csv").onclick = () => download("annotations.csv"); $("download-json").onclick = () => download("metadata.json");
  $("sign-out").onclick = async () => {
    clearTimeout(saveTimer);
    try { await saveDraft(); await queueTick(); } catch(_) { /* Unsynced edits remain in sessionStorage. */ }
    try { await post("/api/auth/logout"); } catch(_) { /* Always allow an expired/offline session to sign out locally. */ }
    stopClock(); token = ""; study = null; sample = null; sessionStorage.removeItem("survey-token"); setAccount(); screen("login");
    $("code-form").hidden = true; $("email-form").hidden = false; $("code").value = "";
  };
  function focusChange() {
    if (!sample || loading || submitting || leaseLost) return;
    if (document.hidden || !document.hasFocus()) {
      if (clockRunning) queueTick(true).catch(() => {});
      clockRunning = false; saveDraft().catch(() => {});
    } else { lastClock = performance.now(); clockRunning = true; }
  }
  document.addEventListener("visibilitychange", focusChange); window.addEventListener("blur", focusChange); window.addEventListener("focus", focusChange);
  window.addEventListener("pagehide", () => {if (clockRunning) queueTick(true).catch(() => {});});
  window.addEventListener("online", () => {if (changed) saveDraft().catch(e => announce(e.message));});
  window.addEventListener("beforeunload", e => { if(changed) {e.preventDefault(); e.returnValue = "";} });
  if (token) api("/api/study").then(result => {study = result; prepare();}).catch(e => {
    if (e.status === 401) {token = ""; sessionStorage.removeItem("survey-token");}
    announce(e.message); setAccount();
  });
})();
