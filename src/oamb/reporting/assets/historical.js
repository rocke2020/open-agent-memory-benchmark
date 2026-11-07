"use strict";
(async () => {
  const app = document.getElementById("app");
  const node = (tag, text, cls) => {
    const element = document.createElement(tag);
    if (text !== undefined && text !== null) element.textContent = String(text);
    if (cls) element.className = cls;
    return element;
  };
  const text = value => typeof value === "string" ? value : JSON.stringify(value, null, 2);
  const fmt = value => Number(value).toLocaleString("en-US", {maximumFractionDigits: 0});
  const pct = value => (value * 100).toFixed(2) + "%";
  try {
    if (typeof DecompressionStream !== "function") throw new Error("This browser does not support offline gzip decoding. Open this report in a current Chrome, Edge, Firefox or Safari, or read README.md and the JSON evidence.");
    const encoded = document.getElementById("evidence").textContent.trim();
    const binary = atob(encoded);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
    const data = JSON.parse(await new Response(stream).text());
    const isBeam = data.dataset === "beam10m";
    const names = Object.keys(data.arms);
    const label = name => name === "oam" ? "OAM" : "Hindsight · saved-context replay";
    const primary = data.arms.oam.results;
    const maps = Object.fromEntries(names.map(name => [name, new Map(data.arms[name].results.map(q => [q.question_id, q]))]));
    const params = new URLSearchParams(location.hash.slice(1));
    const state = {view: params.get("view") === "first_pass" ? "first_pass" : "policy_final", id: params.get("id"), search: "", outcome: "all", category: ""};
    const attempt = q => state.view === "first_pass" ? q.first_pass : q.retry || q.first_pass;
    const passed = q => attempt(q).score >= (isBeam ? 0.5 : 1);
    const statusClass = q => !passed(q) ? "fail" : attempt(q).score < 1 ? "partial" : "pass";
    const filtered = () => primary.filter(q => {
      const score = attempt(q).score;
      if (state.category && state.category !== q.question_type) return false;
      if (state.search && !(q.question_id + " " + q.question).toLowerCase().includes(state.search)) return false;
      if (state.outcome === "fail" && passed(q)) return false;
      if (state.outcome === "pass" && !passed(q)) return false;
      if (state.outcome === "partial" && !(score >= .5 && score < 1)) return false;
      if (state.outcome === "retried" && !q.retry) return false;
      if (state.outcome === "regressed" && !(q.retry && q.retry.score < q.first_pass.score)) return false;
      if (state.outcome === "any_fail" && names.every(name => passed(maps[name].get(q.question_id)))) return false;
      return true;
    });
    const aside = node("aside");
    const head = node("div", null, "side-head");
    const categories = node("details", null, "categories");
    categories.open = true;
    categories.append(node("summary", "Question categories"));
    const categoryList = node("div", null, "category-list");
    categories.append(categoryList);
    const filters = node("div", null, "filters");
    const search = node("input");
    search.id = "search"; search.type = "search"; search.placeholder = "Search questions or IDs…";
    search.setAttribute("aria-label", "Search questions or IDs");
    const outcome = node("select"); outcome.id = "outcome"; outcome.setAttribute("aria-label", "Outcome filter");
    const options = [["all", "All questions"], ["fail", isBeam ? "OAM below threshold" : "Incorrect"], ["pass", isBeam ? "OAM at least 0.5" : "Correct"], ["retried", "With recorded retry"], ["regressed", "Retry score decreased"]];
    if (isBeam) options.push(["partial", "OAM partial score (0.5–1)"], ["any_fail", "Either arm below threshold"]);
    for (const [value, title] of options) { const option = node("option", title); option.value = value; outcome.append(option); }
    const counter = node("div", null, "filter-count"); counter.setAttribute("aria-live", "polite");
    filters.append(search, outcome, counter);
    const list = node("div", null, "question-list"); list.id = "question-list";
    aside.append(head, categories, filters, list);
    const main = node("main"); main.id = "question-detail";
    app.replaceChildren(aside, main);

    function syncHash() {
      const hash = new URLSearchParams({id: state.id || "", view: state.view});
      history.replaceState(null, "", "#" + hash.toString());
    }
    function renderHead() {
      const summary = data.arms.oam.summary[state.view];
      head.replaceChildren(node("p", "OAM · " + (isBeam ? "BEAM-10M" : "LME-500"), "eyebrow"), node("h1", pct(summary.mean)));
      head.append(node("p", isBeam ? "Mean rubric score" : "Answer accuracy", "primary-caption"));
      head.append(node("span", `${summary.passes} / ${summary.count} ${isBeam ? "at least 0.5" : "correct"}`, "secondary"));
      if (isBeam) head.append(node("p", `${pct(summary.passes / summary.count)} threshold pass rate`, "secondary"));
      if (isBeam) {
        const replay = data.arms["hindsight-saved-context-replay"].summary[state.view];
        head.append(node("p", `Replay ${pct(replay.mean)} · ${replay.passes}/${replay.count} at least 0.5`, "secondary"));
      }
      const metrics = node("div", null, "metrics");
      for (const [title, value] of [["Context · mean tokens", fmt(summary.context_mean)], ["Saved recall · median", fmt(summary.retrieval_median_ms) + " ms"], ["Context · total tokens", fmt(summary.context_total)], ["Recorded retries", summary.retries]]) {
        const box = node("div", null, "metric"); box.append(node("span", title), node("strong", value)); metrics.append(box);
      }
      head.append(metrics);
      const modeLabel = node("label", "Show", "mode-label"); modeLabel.htmlFor = "view";
      const mode = node("select"); mode.id = "view"; mode.setAttribute("aria-label", "Outcome policy");
      for (const [value, title] of [["policy_final", "Policy final"], ["first_pass", "First pass"]]) { const option = node("option", title); option.value = value; mode.append(option); }
      mode.value = state.view;
      mode.addEventListener("change", () => { state.view = mode.value; renderHead(); renderCategories(); refresh(); });
      modeLabel.append(mode); head.append(modeLabel);
    }
    function renderCategories() {
      categoryList.replaceChildren();
      const grouped = new Map();
      for (const q of primary) {
        const group = grouped.get(q.question_type) || {count: 0, sum: 0};
        group.count++; group.sum += attempt(q).score; grouped.set(q.question_type, group);
      }
      for (const [name, stats] of grouped) {
        const button = node("button", null, "category-row" + (state.category === name ? " active" : ""));
        button.setAttribute("aria-pressed", String(state.category === name));
        button.append(node("span", name.replaceAll("_", " ")), node("span", stats.count), node("strong", (100 * stats.sum / stats.count).toFixed(1) + "%"));
        button.title = (isBeam ? "Mean rubric score" : "Accuracy") + " · click to filter";
        button.addEventListener("click", () => { state.category = state.category === name ? "" : name; renderCategories(); refresh(); });
        categoryList.append(button);
      }
    }
    function refresh() {
      const rows = filtered();
      if (!rows.some(q => q.question_id === state.id)) state.id = rows[0]?.question_id || null;
      counter.replaceChildren(node("span", `${rows.length} / ${primary.length} questions`));
      if (state.category) { const clear = node("button", "Clear category"); clear.addEventListener("click", () => { state.category = ""; renderCategories(); refresh(); }); counter.append(clear); }
      list.replaceChildren();
      if (!rows.length) list.append(node("p", "No matching questions", "empty"));
      for (const q of rows) {
        const button = node("button", null, "question-row");
        button.dataset.id = q.question_id;
        button.setAttribute("aria-current", String(q.question_id === state.id));
        const line = node("span", null, "question-line");
        line.append(node("strong", passed(q) ? "✓" : "✗", statusClass(q)), node("span", q.question, "question-preview"));
        button.append(line, node("span", `${q.question_id} · ${q.question_type.replaceAll("_", " ")}`, "question-meta"));
        button.addEventListener("click", () => { state.id = q.question_id; refresh(); });
        list.append(button);
      }
      renderDetail(rows);
      syncHash();
    }
    function navigate(direction) {
      const rows = filtered();
      const index = rows.findIndex(q => q.question_id === state.id);
      if (index + direction >= 0 && index + direction < rows.length) {
        state.id = rows[index + direction].question_id; refresh();
        Array.from(list.children).find(el => el.dataset.id === state.id)?.scrollIntoView({block: "nearest"});
      }
    }
    function armCard(q, name) {
      const current = attempt(q);
      const card = node("div", null, "arm"); card.dataset.arm = name;
      const heading = node("div", null, "arm-title");
      heading.append(node("h2", label(name)), node("span", pct(current.score), "arm-quality " + statusClass(q)));
      card.append(heading);
      const panel = node("section", null, "answer-panel");
      panel.append(node("p", "System answer", "eyebrow"), node("div", current.answer, "answer-text"));
      panel.append(node("p", isBeam ? (passed(q) ? "At least 0.5" : "Below 0.5") : (passed(q) ? "Correct" : "Incorrect"), "verdict " + statusClass(q)));
      panel.append(node("p", `Answer ${current.answer_model || "recorded model"} · ${current.answer_effort}; judge ${current.judge_model || "recorded model"} · ${current.judge_effort}`, "binding"));
      if (current.judgments?.length) {
        const details = node("details"); details.append(node("summary", `Rubric judgments · ${current.judgments.length} criteria`));
        const items = node("ol", null, "judge-list");
        for (const judgment of current.judgments) { const item = node("li"); item.append(node("strong", `${judgment.score.toFixed(1)} · ${judgment.criterion}`), node("p", judgment.reason)); items.append(item); }
        details.append(items); panel.append(details);
      } else panel.append(node("p", current.judge_reason || "See saved judgment in source evidence.", "secondary"));
      if (current.metadata_projection) panel.append(node("p", "Recorded metadata-only projection; original schema-invalid visible response is retained in the source archive.", "secondary"));
      card.append(panel);
      const context = node("section", null, "context-panel");
      const details = node("details"); details.open = true;
      details.append(node("summary", "Injected context · " + fmt(q.context_tokens) + " tokens"));
      details.append(node("p", `${q.context_tokenizer} · saved recall ${fmt(q.retrieve_time_ms)} ms`, "context-stats"));
      const pre = node("pre", q.context); pre.dataset.context = q.question_id; details.append(pre);
      details.append(node("p", "SHA-256 " + q.context_sha256, "context-stats")); context.append(details); card.append(context);
      const history = node("section", null, "attempt-history");
      const attempts = node("details"); attempts.append(node("summary", q.retry ? "First pass and max-effort retry" : "First-pass record · no retry"));
      for (const [key, title] of [["first_pass", "First pass"], ["retry", "Max-effort retry"]]) {
        const saved = q[key]; if (!saved) continue;
        const article = node("article"); article.append(node("strong", `${title} · ${pct(saved.score)}`));
        if ((q.policy_final_attempt === "retry" ? key === "retry" : key === "first_pass")) article.append(node("span", "Policy final", "badge"));
        if (key === "retry" && saved.score < q.first_pass.score) article.append(node("span", "Recorded score regression", "badge fail"));
        article.append(node("pre", saved.answer)); attempts.append(article);
      }
      history.append(attempts); card.append(history);
      const source = node("div", null, "source-panel");
      const link = node("a", "Full saved requests, responses and provenance (.json.gz) ↗"); link.href = q.source_file; source.append(link);
      source.append(node("span", "Original result SHA-256 " + q.source.record_sha256)); card.append(source);
      return card;
    }
    function renderDetail(rows) {
      const inner = node("div", null, "main-inner"); main.replaceChildren(inner); main.scrollTop = 0;
      if (!state.id) { inner.append(node("p", "No matching questions. Change the filters to continue.", "empty")); return; }
      const q = maps.oam.get(state.id);
      const index = rows.findIndex(item => item.question_id === state.id);
      const nav = node("div", null, "navigator");
      const prev = node("button", "← Previous"); prev.disabled = index <= 0; prev.addEventListener("click", () => navigate(-1));
      const next = node("button", "Next →"); next.disabled = index >= rows.length - 1; next.addEventListener("click", () => navigate(1));
      nav.append(prev, node("span", `${index + 1} / ${rows.length} · ${state.view === "first_pass" ? "First pass" : "Policy final"}`), next); inner.append(nav);
      const queryLabel = node("div", null, "query-label");
      queryLabel.append(node("strong", "QUESTION"), node("span", q.question_id));
      const permalink = node("a", "Link to question ↗"); permalink.href = "#" + new URLSearchParams({id: q.question_id, view: state.view}); queryLabel.append(permalink);
      inner.append(queryLabel, node("section", q.question, "query-panel"));
      const reference = node("section", null, "reference"); reference.append(node("p", "Gold answer", "eyebrow"));
      reference.append(node("p", q.gold_answers.length ? q.gold_answers.map(text).join("\n\n") : "No standalone gold answer; evaluate against the original rubric below."));
      if (q.rubric?.length) { const details = node("details"); details.append(node("summary", `Original rubric · ${q.rubric.length} criteria`)); const ol = node("ol"); for (const item of q.rubric) ol.append(node("li", text(item))); details.append(ol); reference.append(details); }
      inner.append(reference);
      const arms = node("div", null, "arms" + (names.length === 1 ? " single" : ""));
      for (const name of names) arms.append(armCard(maps[name].get(state.id), name));
      inner.append(arms);
      const notice = node("footer", null, "notice");
      notice.append(node("p", isBeam ? "Historical OAM L1 evaluation compared with Hindsight saved-context replay. The replay reuses historical contexts; it is not a matched rerun of Hindsight indexing or retrieval. Mean rubric score and threshold pass rate are separate metrics." : "Historical OAM development campaign, consolidated from six disjoint contributions. First-pass and recorded retry outcomes remain separate; policy final uses the dispatched retry even when its score falls."));
      notice.append(node("p", "Context and latency statistics use the full saved question cohort. Tokenizers and saved runtime conditions differ between evaluations. Missing billing and indexing measurements remain unavailable."));
      const metadata = node("details", null, "metadata"); metadata.append(node("summary", "Dataset, runtime and evidence provenance"), node("pre", JSON.stringify(data.manifest, null, 2))); notice.append(metadata);
      const design = node("p"); const designLink = node("a", "AMB report design reference"); designLink.href = "https://github.com/vectorize-io/agent-memory-benchmark"; design.append(designLink); notice.append(design); inner.append(notice);
    }
    search.addEventListener("input", () => { state.search = search.value.trim().toLowerCase(); refresh(); });
    outcome.addEventListener("change", () => { state.outcome = outcome.value; refresh(); });
    document.addEventListener("keydown", event => {
      if (event.ctrlKey || event.metaKey || event.altKey || event.target.closest("input,textarea,select")) return;
      if (["ArrowDown", "j", "ArrowUp", "k"].includes(event.key)) { event.preventDefault(); navigate(["ArrowDown", "j"].includes(event.key) ? 1 : -1); }
    });
    window.addEventListener("hashchange", () => {
      const hash = new URLSearchParams(location.hash.slice(1));
      if (maps.oam.has(hash.get("id"))) state.id = hash.get("id");
      state.view = hash.get("view") === "first_pass" ? "first_pass" : "policy_final";
      renderHead(); renderCategories(); refresh();
    });
    renderHead(); renderCategories(); refresh();
    document.documentElement.dataset.reportReady = "true";
  } catch (error) {
    app.replaceChildren(node("p", "Could not open evidence: " + error.message, "empty"));
    document.documentElement.dataset.reportError = "true";
  }
})();
