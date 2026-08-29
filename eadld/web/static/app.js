const form = document.querySelector("#seed-form");
const button = document.querySelector("#generate");
const exportButton = document.querySelector("#export-seq");
const quota = document.querySelector("#quota");
const message = document.querySelector("#message");
const metrics = document.querySelector("#metrics");
const candidateBar = document.querySelector("#candidate-bar");
const candidateCount = document.querySelector("#candidate-count");
const candidateButtons = document.querySelector("#candidate-buttons");
const images = {
  layout: document.querySelector("#layout"),
  spots: document.querySelector("#spots"),
};
let objectUrls = [];
let seqUrl = null;
let candidates = [];
let remainingQuota = null;

function optionalNumber(data, name) {
  const value = data.get(name)?.toString().trim();
  return value ? Number(value) : null;
}

function setBusy(busy) {
  button.disabled = busy || remainingQuota === 0;
  exportButton.disabled = busy || !seqUrl;
  button.querySelector("span").textContent = busy ? "生成并追迹中" : "生成初始结构";
}

function showMessage(text, error = false) {
  message.textContent = text;
  message.classList.toggle("error", error);
}

async function readError(response) {
  try {
    const body = await response.json();
    return body.detail || "请求失败";
  } catch {
    return "请求失败";
  }
}

async function loadProtectedImage(url) {
  const response = await fetch(url, {
    headers: { "X-EADLD-Request": "1" },
    cache: "no-store",
  });
  if (!response.ok) throw new Error(await readError(response));
  const objectUrl = URL.createObjectURL(await response.blob());
  objectUrls.push(objectUrl);
  return objectUrl;
}

async function showCandidate(candidate) {
  seqUrl = candidate.files.seq;
  exportButton.disabled = true;
  objectUrls.forEach((url) => URL.revokeObjectURL(url));
  objectUrls = [];
  const [layoutUrl, spotsUrl] = await Promise.all([
    loadProtectedImage(candidate.images.layout),
    loadProtectedImage(candidate.images.spots),
  ]);
  images.layout.src = layoutUrl;
  images.spots.src = spotsUrl;
  Object.values(images).forEach((image) => {
    image.hidden = false;
    image.previousElementSibling.hidden = true;
  });
  Object.entries(candidate.metrics).forEach(([key, value]) => {
    const target = metrics.querySelector(`[data-key="${key}"]`);
    if (target) target.textContent = value;
  });
  candidateButtons.querySelectorAll("button").forEach((item) => {
    item.classList.toggle("active", Number(item.dataset.rank) === candidate.rank);
  });
  metrics.hidden = false;
  exportButton.disabled = false;
}

function renderCandidateButtons() {
  candidateButtons.replaceChildren();
  candidates.forEach((candidate) => {
    const item = document.createElement("button");
    item.type = "button";
    item.dataset.rank = candidate.rank;
    item.textContent = candidate.rank;
    item.setAttribute("aria-label", `查看候选 ${candidate.rank}`);
    item.addEventListener("click", async () => {
      try {
        await showCandidate(candidate);
        showMessage(`候选 ${candidate.rank}`);
      } catch (error) {
        showMessage(error.message, true);
      }
    });
    candidateButtons.append(item);
  });
}

exportButton.addEventListener("click", async () => {
  if (!seqUrl) return;
  try {
    const response = await fetch(seqUrl, {
      headers: { "X-EADLD-Request": "1" },
      cache: "no-store",
    });
    if (!response.ok) throw new Error(await readError(response));
    const url = URL.createObjectURL(await response.blob());
    const link = document.createElement("a");
    link.href = url;
    link.download = "eadld_initial_structure.seq";
    link.click();
    URL.revokeObjectURL(url);
    showMessage("SEQ 已导出");
  } catch (error) {
    showMessage(error.message, true);
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  seqUrl = null;
  candidateBar.hidden = true;
  setBusy(true);
  showMessage("网络生成结构，光学引擎正在追迹…");

  const data = new FormData(form);
  const payload = {
    efl_mm: Number(data.get("efl_mm")),
    f_number: Number(data.get("f_number")),
    half_field_deg: Number(data.get("half_field_deg")),
    wavelengths_nm: data.get("wavelengths_nm").toString().split(/[\s,;]+/).filter(Boolean).map(Number),
    elements: Number(data.get("elements")),
    candidate_count: Number(data.get("candidate_count")),
    min_image_clearance_mm: optionalNumber(data, "min_image_clearance_mm"),
    max_package_length_mm: optionalNumber(data, "max_package_length_mm"),
    max_distortion_fraction: optionalNumber(data, "max_distortion_fraction"),
    target_cra_deg: optionalNumber(data, "target_cra_deg"),
  };

  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-EADLD-Request": "1",
      },
      cache: "no-store",
      body: JSON.stringify(payload),
    });
    if (!response.ok) throw new Error(await readError(response));
    const result = await response.json();
    candidates = result.candidates?.length ? result.candidates : [{
      rank: 1,
      metrics: result.metrics,
      images: result.images,
      files: result.files,
    }];
    candidateCount.textContent = `${result.returned_candidates || candidates.length} / ${result.requested_candidates || candidates.length} 个候选`;
    renderCandidateButtons();
    candidateBar.hidden = false;
    await showCandidate(candidates[0]);
    if (result.quota) {
      remainingQuota = result.quota.remaining;
      quota.textContent = `今日还可生成 ${remainingQuota} 次`;
    }
    const complete = (result.returned_candidates || candidates.length) === (result.requested_candidates || candidates.length);
    showMessage(complete ? "生成完成" : "已返回通过门槛的候选");
  } catch (error) {
    showMessage(error.message, true);
    refreshQuota();
  } finally {
    setBusy(false);
  }
});

async function refreshQuota() {
  try {
    const response = await fetch("/api/quota", { cache: "no-store" });
    if (!response.ok) return;
    const result = await response.json();
    remainingQuota = result.remaining;
    quota.textContent = `今日还可生成 ${remainingQuota} 次`;
    button.disabled = remainingQuota < 1;
  } catch {
    quota.textContent = "";
  }
}

refreshQuota();
