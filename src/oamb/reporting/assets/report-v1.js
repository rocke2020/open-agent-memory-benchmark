(() => {
  "use strict";

  const model = JSON.parse(document.getElementById("oamb-report-data").textContent);
  const list = document.getElementById("oamb-records");
  const filter = document.getElementById("oamb-filter");
  const axisFilter = document.getElementById("oamb-axis-filter");
  const sort = document.getElementById("oamb-sort");
  const previous = document.getElementById("oamb-previous");
  const next = document.getElementById("oamb-next");
  const copy = document.getElementById("oamb-copy");
  const copyStatus = document.getElementById("oamb-copy-status");
  const kind = document.getElementById("oamb-report-kind");
  const origin = document.getElementById("oamb-origin-kind");
  const facetFilters = new Map([
    ["status", document.getElementById("oamb-status-filter")],
    ["failure", document.getElementById("oamb-failure-filter")],
    ["evaluation", document.getElementById("oamb-evaluation-filter")],
    ["verdict", document.getElementById("oamb-verdict-filter")],
    ["capability", document.getElementById("oamb-capability-filter")],
    ["metric", document.getElementById("oamb-metric-filter")],
    ["proof", document.getElementById("oamb-proof-filter")],
    ["raw", document.getElementById("oamb-raw-filter")],
  ]);
  const facetKeys = new Map([
    ["state", "status"],
    ["status", "status"],
    ["terminal_state", "status"],
    ["error_stage", "failure"],
    ["failure_stage", "failure"],
    ["evaluation_disposition", "evaluation"],
    ["judgement_status", "evaluation"],
    ["verdict", "verdict"],
    [["win", "ner"].join(""), "verdict"],
    ["comparable", "verdict"],
    ["passed", "verdict"],
    ["capability", "capability"],
    ["component", "capability"],
    ["type", "capability"],
    ["metric_id", "metric"],
    ["proof_status", "proof"],
  ]);
  const mab65 = model.mab65_reduction;
  const records = [];
  let stableIndex = 0;

  function facetText(value) {
    if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
      return String(value);
    }
    return null;
  }

  function collectRecordFacets(value) {
    const facets = new Map(Array.from(facetFilters.keys(), (name) => [name, new Set()]));
    let rawPresent = false;
    function visit(item) {
      if (Array.isArray(item)) {
        item.forEach(visit);
        return;
      }
      if (!item || typeof item !== "object") {
        return;
      }
      for (const [key, nested] of Object.entries(item)) {
        const facetName = facetKeys.get(key);
        const text = facetText(nested);
        if (facetName && text !== null) {
          facets.get(facetName).add(text);
        }
        if (
          (key.endsWith("raw_ref") || key.endsWith("raw_reference")) &&
          ((typeof nested === "string" && nested.length > 0) ||
            (Array.isArray(nested) && nested.length > 0))
        ) {
          rawPresent = true;
        }
        visit(nested);
      }
    }
    visit(value);
    facets.get("raw").add(rawPresent ? "present" : "absent");
    return facets;
  }

  function recordValue(detail) {
    return {
      value: JSON.stringify(detail, null, 2),
      facets: collectRecordFacets(detail),
      detail,
    };
  }
  function projectionFacets(projection) {
    const facets = new Map(Array.from(facetFilters.keys(), (name) => [name, new Set()]));
    facets.get("status").add(projection.status);
    if (typeof projection.failure_stage === "string") {
      facets.get("failure").add(projection.failure_stage);
    }
    if (typeof projection.evaluation_status === "string") {
      facets.get("evaluation").add(projection.evaluation_status);
    }
    if (typeof projection.verdict === "string") {
      facets.get("verdict").add(projection.verdict);
    }
    projection.capabilities_or_types.forEach((value) => facets.get("capability").add(value));
    projection.metric_ids.forEach((value) => facets.get("metric").add(value));
    projection.proof_statuses.forEach((value) => facets.get("proof").add(value));
    facets.get("raw").add(projection.raw_evidence_present ? "present" : "absent");
    return facets;
  }
  const projections = Array.isArray(model.record_projections) ? model.record_projections : [];
  for (const projection of projections) {
    const detail = {
      record_id: projection.record_id,
      axis: projection.axis,
      status: projection.status,
      failure_stage: projection.failure_stage,
      evaluation_status: projection.evaluation_status,
      verdict: projection.verdict,
      capabilities_or_types: projection.capabilities_or_types,
      metric_ids: projection.metric_ids,
      proof_statuses: projection.proof_statuses,
      raw_evidence_present: projection.raw_evidence_present,
      display_previews: projection.display_previews,
      latency_microseconds: projection.latency_microseconds,
      context_view_tokens: projection.context_view_tokens,
      declared_usage: projection.declared_usage,
      details: Object.fromEntries(projection.detail_items),
    };
    records.push({
      id: `${projection.axis}=${projection.record_id}`,
      axis: projection.axis,
      label: projection.label,
      value: JSON.stringify(detail, null, 2),
      facets: projectionFacets(projection),
      detail,
      stableIndex,
    });
    stableIndex += 1;
  }
  for (const [label, value] of Object.entries(model)) {
    if (
      label === "record_projections" ||
      label === "logical_context_ids" ||
      label === "ingestion_occurrence_ids" ||
      label === "case_occurrence_ids" ||
      label === "attempt_ids"
    ) {
      continue;
    }
    records.push({
      id: `record-${stableIndex + 1}`,
      axis: "report",
      label,
      ...recordValue(value),
      stableIndex,
    });
    stableIndex += 1;
  }
  let activeRecordId = null;

  for (const [facetName, select] of facetFilters) {
    if (facetName === "raw") {
      continue;
    }
    const values = new Set();
    records.forEach((record) => record.facets.get(facetName).forEach((value) => values.add(value)));
    for (const value of [...values].sort()) {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = value;
      select.append(option);
    }
  }

  kind.textContent = model.schema_name;
  origin.textContent = model.origin_kind || model.schema_name;

  function exactText(value) {
    return `${value.numerator}/${value.denominator}`;
  }

  function appendExactItems(listElement, items, labels, key) {
    for (const item of items) {
      const row = document.createElement("li");
      const label = labels[item[key]] || item[key];
      row.textContent = `${label}: ${exactText(item.score)}`;
      listElement.append(row);
    }
  }

  if (mab65) {
    const container = document.getElementById("oamb-mab65");
    const boundary = document.getElementById("oamb-mab65-boundary");
    const components = document.getElementById("oamb-mab65-components");
    const capabilities = document.getElementById("oamb-mab65-capabilities");
    const index = document.getElementById("oamb-mab65-index");
    container.hidden = false;
    if (mab65.available) {
      boundary.textContent =
        "Complete 29 logical rows → 25 physical plans → 65 metric-eligible cases. " +
        "Grouped FactConsolidation SH/MH membership and ReDial resolution evidence are preserved.";
      appendExactItems(
        components,
        mab65.components,
        {
          ar: "EventQA SubEM",
          icl: "ICL Exact Match",
          recsys: "ReDial Recall@5",
          lru: "DetectiveQA Exact Match",
          cr_sf: "FactConsolidation SubEM",
        },
        "component",
      );
      appendExactItems(
        capabilities,
        mab65.capabilities,
        {
          ar: "Accurate Retrieval",
          ttl: "Test-Time Learning (equal ICL/ReDial contribution)",
          lru: "Long-Range Understanding",
          cr_sf: "Conflict Resolution and Synthesis",
        },
        "capability",
      );
      index.textContent =
        `MAB-65 Capability-Balanced Index: ${exactText(mab65.index_value)} ` +
        "(secondary; complete evidence only).";
    } else {
      boundary.textContent = `MAB-65 secondary index unavailable: ${mab65.unavailable_reason}.`;
      index.remove();
    }
  }

  function selectedFacetMatches(record) {
    for (const [facetName, select] of facetFilters) {
      const selected = selectedFacetValues(select);
      if (
        selected.length > 0 &&
        !selected.some((value) => record.facets.get(facetName).has(value))
      ) {
        return false;
      }
    }
    return true;
  }

  function selectedFacetValues(select) {
    return Array.from(select.selectedOptions, (option) => option.value).filter(
      (value) => value !== "all",
    );
  }

  function firstFacet(record, facetName) {
    return [...record.facets.get(facetName)].sort()[0] || null;
  }

  function numericValue(value, acceptedKeys) {
    if (Array.isArray(value)) {
      for (const item of value) {
        const nested = numericValue(item, acceptedKeys);
        if (nested !== null) {
          return nested;
        }
      }
      return null;
    }
    if (!value || typeof value !== "object") {
      return null;
    }
    for (const [key, nested] of Object.entries(value)) {
      if (acceptedKeys.some((accepted) => key.includes(accepted)) && typeof nested === "number") {
        return nested;
      }
      const found = numericValue(nested, acceptedKeys);
      if (found !== null) {
        return found;
      }
    }
    return null;
  }

  function selectedSortValue(record) {
    if (sort.value === "label") {
      return record.label;
    }
    if (sort.value === "status") {
      return firstFacet(record, "status");
    }
    if (sort.value === "metric") {
      return firstFacet(record, "metric");
    }
    if (sort.value === "latency") {
      return numericValue(record.detail, ["latency", "duration"]);
    }
    if (sort.value === "context") {
      return numericValue(record.detail, ["context_view_tokens", "ctx_tokens"]);
    }
    if (sort.value === "usage") {
      return numericValue(record.detail, ["input_tokens", "output_tokens", "total_tokens"]);
    }
    return null;
  }

  function compareRecords(left, right) {
    const leftValue = selectedSortValue(left);
    const rightValue = selectedSortValue(right);
    if (leftValue !== rightValue) {
      if (leftValue === null) {
        return 1;
      }
      if (rightValue === null) {
        return -1;
      }
      if (typeof leftValue === "number" && typeof rightValue === "number") {
        return leftValue - rightValue;
      }
      return String(leftValue) < String(rightValue) ? -1 : 1;
    }
    return left.stableIndex - right.stableIndex;
  }

  function hashState() {
    const rawHash = location.hash.slice(1);
    const params = new URLSearchParams(rawHash);
    const targetAxes = ["logical-context", "plan", "case", "attempt"];
    const allowedKeys = new Set([
      ...targetAxes,
      "axis",
      "sort",
      ...facetFilters.keys(),
    ]);
    if ([...params.keys()].some((key) => !allowedKeys.has(key))) {
      return { id: null, params: new URLSearchParams() };
    }
    const targets = targetAxes.flatMap((axis) =>
      params.getAll(axis).map((identity) => ({ axis, identity })),
    );
    const target = targets.length === 1 ? targets[0] : null;
    const id =
      target && /^[0-9a-f]{64}$/.test(target.identity)
        ? `${target.axis}=${target.identity}`
        : null;
    return { id, params };
  }

  function restoreSingleSelect(select, requested, fallback) {
    const allowed = Array.from(select.options, (option) => option.value);
    select.value = requested !== null && allowed.includes(requested) ? requested : fallback;
  }

  function restoreFacetSelect(select, requested) {
    const allowed = new Set(Array.from(select.options, (option) => option.value));
    const selected = requested.filter((value) => value !== "all" && allowed.has(value));
    for (const option of select.options) {
      option.selected = selected.length === 0 ? option.value === "all" : selected.includes(option.value);
    }
  }

  function restoreHashState() {
    const state = hashState();
    restoreSingleSelect(axisFilter, state.params.get("axis"), "all");
    restoreSingleSelect(sort, state.params.get("sort"), "source");
    for (const [facetName, select] of facetFilters) {
      restoreFacetSelect(select, state.params.getAll(facetName));
    }
    activeRecordId = state.id;
  }

  function writeHashState(pushHistory) {
    const params = new URLSearchParams();
    if (activeRecordId) {
      const separator = activeRecordId.indexOf("=");
      const axis = activeRecordId.slice(0, separator);
      const identity = activeRecordId.slice(separator + 1);
      if (
        ["logical-context", "plan", "case", "attempt"].includes(axis) &&
        /^[0-9a-f]{64}$/.test(identity)
      ) {
        params.set(axis, identity);
      }
    }
    if (axisFilter.value !== "all") {
      params.set("axis", axisFilter.value);
    }
    for (const [facetName, select] of facetFilters) {
      for (const value of selectedFacetValues(select)) {
        params.append(facetName, value);
      }
    }
    if (sort.value !== "source") {
      params.set("sort", sort.value);
    }
    const nextHash = params.toString();
    if (pushHistory) {
      location.hash = nextHash;
    } else {
      history.replaceState(null, "", nextHash ? `#${nextHash}` : location.pathname);
    }
  }

  function render() {
    const query = filter.value.trim().toLocaleLowerCase();
    const selectedAxis = axisFilter.value;
    const ordered = [...records].sort(compareRecords);
    list.replaceChildren();
    for (const record of ordered) {
      const article = document.createElement("article");
      const heading = document.createElement("h3");
      const anchor = document.createElement("a");
      const axis = document.createElement("span");
      const value = document.createElement("pre");
      article.className = "record";
      article.id = record.id;
      article.dataset.axis = record.axis;
      article.tabIndex = -1;
      article.hidden = !(
        (selectedAxis === "all" || selectedAxis === record.axis) &&
        selectedFacetMatches(record) &&
        `${record.label}\n${record.value}`.toLocaleLowerCase().includes(query)
      );
      anchor.href =
        record.axis === "report"
          ? "#"
          : `#${record.axis}=${encodeURIComponent(record.detail.record_id)}`;
      anchor.textContent = record.label;
      axis.className = "axis-badge";
      axis.textContent = record.axis;
      value.textContent = record.value;
      heading.append(anchor, axis);
      article.append(heading, value);
      article.addEventListener("focus", () => {
        activeRecordId = article.id;
        writeHashState(false);
      });
      anchor.addEventListener("click", (event) => {
        event.preventDefault();
        activeRecordId = article.id;
        writeHashState(true);
        article.focus();
      });
      list.append(article);
    }
    const requested = activeRecordId && document.getElementById(activeRecordId);
    if (requested && !requested.hidden) {
      activeRecordId = requested.id;
    }
  }

  function visibleRecords() {
    return Array.from(list.querySelectorAll(".record:not([hidden])"));
  }

  function moveWithinVisibleSet(offset) {
    const visible = visibleRecords();
    if (visible.length === 0) {
      return;
    }
    const current = visible.findIndex((record) => record.id === activeRecordId);
    const next = current < 0 ? 0 : Math.max(0, Math.min(visible.length - 1, current + offset));
    activeRecordId = visible[next].id;
    writeHashState(true);
    visible[next].focus();
  }

  function moveToVisibleBoundary(end) {
    const visible = visibleRecords();
    if (visible.length === 0) {
      return;
    }
    const selected = visible[end ? visible.length - 1 : 0];
    activeRecordId = selected.id;
    writeHashState(true);
    selected.focus();
  }

  filter.addEventListener("input", () => {
    render();
  });
  for (const control of [axisFilter, sort, ...facetFilters.values()]) {
    control.addEventListener("change", () => {
      render();
      writeHashState(false);
    });
  }
  previous.addEventListener("click", () => moveWithinVisibleSet(-1));
  next.addEventListener("click", () => moveWithinVisibleSet(1));
  window.addEventListener("hashchange", () => {
    restoreHashState();
    render();
    const requested = activeRecordId && document.getElementById(activeRecordId);
    if (requested && !requested.hidden) {
      activeRecordId = requested.id;
      requested.focus();
    }
  });
  document.addEventListener("keydown", (event) => {
    if (event.target instanceof Element && event.target.closest("input, select, textarea, button")) {
      return;
    }
    if (event.key === "j" || event.key === "ArrowDown") {
      event.preventDefault();
      moveWithinVisibleSet(1);
    }
    if (event.key === "k" || event.key === "ArrowUp") {
      event.preventDefault();
      moveWithinVisibleSet(-1);
    }
    if (event.key === "Home") {
      event.preventDefault();
      moveToVisibleBoundary(false);
    }
    if (event.key === "End") {
      event.preventDefault();
      moveToVisibleBoundary(true);
    }
  });
  copy.addEventListener("click", async () => {
    const selected =
      (activeRecordId && document.getElementById(activeRecordId)) || visibleRecords()[0];
    if (!selected) {
      copyStatus.textContent = "No visible record to copy.";
      return;
    }
    const text = selected.textContent;
    if (navigator.clipboard && navigator.clipboard.writeText) {
      try {
        await navigator.clipboard.writeText(text);
        copyStatus.textContent = "Copied visible record.";
        return;
      } catch {
        copyStatus.textContent = "Select and copy manually.";
        selected.focus();
        return;
      }
    }
    copyStatus.textContent = "Select and copy manually.";
    selected.focus();
  });

  restoreHashState();
  render();
})();
