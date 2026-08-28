const form = document.querySelector("#seed-form");
const button = document.querySelector("#generate");
const message = document.querySelector("#message");
const metrics = document.querySelector("#metrics");
const images = {
  layout: document.querySelector("#layout"),
  spots: document.querySelector("#spots"),
};
let objectUrls = [];

function optionalNumber(data, name) {
  const value = data.get(name)?.toString().trim();
  return value ? Number(value) : null;
}

function setBusy(busy) {
  button.disabled = busy;
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

async function loadProtectedImage(url, token) {
  const response = await fetch(url, {
    headers: { Authorization: `Bearer ${token}`, "X-EADLD-Request": "1" },
    cache: "no-store",
  });
  if (!response.ok) throw new Error(await readError(response));
  const objectUrl = URL.createObjectURL(await response.blob());
  objectUrls.push(objectUrl);
  return objectUrl;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  setBusy(true);
  showMessage("网络生成结构，光学引擎正在追迹…");

  const data = new FormData(form);
  const token = data.get("access_token").toString();
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
        Authorization: `Bearer ${token}`,
        "X-EADLD-Request": "1",
      },
      cache: "no-store",
      body: JSON.stringify(payload),
    });
    if (!response.ok) throw new Error(await readError(response));
    const result = await response.json();

    objectUrls.forEach((url) => URL.revokeObjectURL(url));
    objectUrls = [];
    const [layoutUrl, spotsUrl] = await Promise.all([
      loadProtectedImage(result.images.layout, token),
      loadProtectedImage(result.images.spots, token),
    ]);
    images.layout.src = layoutUrl;
    images.spots.src = spotsUrl;
    Object.values(images).forEach((image) => {
      image.hidden = false;
      image.previousElementSibling.hidden = true;
    });
    Object.entries(result.metrics).forEach(([key, value]) => {
      const target = metrics.querySelector(`[data-key="${key}"]`);
      if (target) target.textContent = value;
    });
    metrics.hidden = false;
    showMessage("生成完成");
  } catch (error) {
    showMessage(error.message, true);
  } finally {
    setBusy(false);
  }
});
