from pathlib import Path

from fastapi.testclient import TestClient

from eadld.web.app import WebSettings, create_app


class HiddenBackend:
    pass


def fake_audit(spec, backend, output_dir: Path):
    output_dir.mkdir(parents=True)
    (output_dir / "layout.png").write_bytes(b"\x89PNG\r\nlayout")
    (output_dir / "spots.png").write_bytes(b"\x89PNG\r\nspots")
    return {
        "selected_candidate_id": "never-return-this-id",
        "candidates": [
            {
                "candidate_id": "never-return-this-id",
                "secret_prescription": "never-return-this-prescription",
                "metrics": {
                    "efl_mm": 74.17171,
                    "bfl_mm": 6.7312,
                    "ttl_mm": 54.8021,
                    "mean_rms_radius_um": 9.96672,
                    "worst_rms_radius_um": 12.0932,
                    "valid_ray_fraction": 1.0,
                },
            }
        ],
    }


def make_client(tmp_path, rate_limit=20):
    settings = WebSettings(
        backend_factory=None,
        backend_config=None,
        access_token="a" * 32,
        public_origin="http://testserver",
        allowed_hosts=("testserver",),
        result_root=tmp_path / "results",
        rate_limit_per_minute=rate_limit,
    )
    app = create_app(settings=settings, backend=HiddenBackend(), audit_runner=fake_audit)
    return TestClient(app)


def valid_payload():
    return {
        "efl_mm": 74,
        "f_number": 2.8,
        "half_field_deg": 6.17,
        "wavelengths_nm": [435, 545.5, 656],
        "elements": 9,
        "candidate_count": 3,
        "min_image_clearance_mm": 6.3,
        "max_package_length_mm": 55.5,
        "max_distortion_fraction": 0.01,
        "target_cra_deg": 12,
    }


def headers(token="a" * 32, origin="http://testserver"):
    return {
        "Authorization": f"Bearer {token}",
        "Origin": origin,
        "X-EADLD-Request": "1",
    }


def test_home_has_security_headers(tmp_path):
    response = make_client(tmp_path).get("/")
    assert response.status_code == 200
    assert "default-src 'self'" in response.headers["content-security-policy"]
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["cache-control"] == "no-store"


def test_generation_returns_only_sanitized_metrics_and_protected_images(tmp_path):
    client = make_client(tmp_path)
    response = client.post("/api/generate", json=valid_payload(), headers=headers())
    assert response.status_code == 200
    assert response.json()["metrics"] == {
        "efl_mm": 74.172,
        "bfl_mm": 6.731,
        "ttl_mm": 54.802,
        "mean_rms_um": 9.967,
        "worst_rms_um": 12.093,
        "valid_ray_percent": 100.0,
    }
    assert "never-return" not in response.text
    assert "prescription" not in response.text

    layout_url = response.json()["images"]["layout"]
    assert client.get(layout_url).status_code == 401
    image = client.get(
        layout_url,
        headers={"Authorization": "Bearer " + "a" * 32, "X-EADLD-Request": "1"},
    )
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/png"


def test_rejects_wrong_token_origin_and_extra_fields(tmp_path):
    client = make_client(tmp_path)
    assert client.post("/api/generate", json=valid_payload(), headers=headers(token="wrong")).status_code == 401
    assert client.post(
        "/api/generate", json=valid_payload(), headers=headers(origin="https://attacker.invalid")
    ).status_code == 403
    payload = {**valid_payload(), "backend_factory": "attacker.module:factory"}
    assert client.post("/api/generate", json=payload, headers=headers()).status_code == 422


def test_rate_limit_and_path_validation(tmp_path):
    client = make_client(tmp_path, rate_limit=1)
    assert client.post("/api/generate", json=valid_payload(), headers=headers()).status_code == 200
    assert client.post("/api/generate", json=valid_payload(), headers=headers()).status_code == 429
    assert client.get(
        "/api/results/not-a-job/layout",
        headers={"Authorization": "Bearer " + "a" * 32, "X-EADLD-Request": "1"},
    ).status_code == 404
