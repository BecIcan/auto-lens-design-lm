from pathlib import Path

from fastapi.testclient import TestClient

from eadld.web.app import WebSettings, create_app


class HiddenBackend:
    pass


def fake_audit(spec, backend, output_dir: Path):
    output_dir.mkdir(parents=True)
    (output_dir / "layout.png").write_bytes(b"\x89PNG\r\nlayout")
    (output_dir / "spots.png").write_bytes(b"\x89PNG\r\nspots")
    (output_dir / "initial_structure.seq").write_text("RDM;LEN\nTITLE 'TEST'\nGO\n", encoding="utf-8")
    second = output_dir / "candidate-02"
    second.mkdir()
    (second / "layout.png").write_bytes(b"\x89PNG\r\nlayout-2")
    (second / "spots.png").write_bytes(b"\x89PNG\r\nspots-2")
    (second / "initial_structure.seq").write_text(
        "RDM;LEN\nTITLE 'TEST-2'\nGO\n", encoding="utf-8"
    )
    return {
        "selected_candidate_id": "never-return-this-id",
        "candidates": [
            {
                "candidate_id": "never-return-this-id",
                "rank": 1,
                "elements": spec.elements,
                "passed": True,
                "secret_prescription": "never-return-this-prescription",
                "metrics": {
                    "efl_mm": 74.17171,
                    "bfl_mm": 6.7312,
                    "ttl_mm": 54.8021,
                    "mean_rms_radius_um": 9.96672,
                    "worst_rms_radius_um": 12.0932,
                    "valid_ray_fraction": 1.0,
                },
            },
            {
                "candidate_id": "never-return-this-id-2",
                "rank": 2,
                "elements": spec.elements,
                "secret_prescription": "never-return-this-prescription-2",
                "passed": True,
                "metrics": {
                    "efl_mm": 73.9,
                    "bfl_mm": 7.1,
                    "ttl_mm": 54.2,
                    "mean_rms_radius_um": 11.2,
                    "worst_rms_radius_um": 13.1,
                    "valid_ray_fraction": 0.99,
                },
            },
        ],
    }


def make_client(
    tmp_path,
    rate_limit=20,
    daily_limit=5,
    access_token=None,
    client_ip_header=None,
    allowed_hosts=("testserver",),
):
    settings = WebSettings(
        backend_factory=None,
        backend_config=None,
        access_token=access_token,
        public_origin="http://testserver",
        allowed_hosts=allowed_hosts,
        result_root=tmp_path / "results",
        rate_limit_per_minute=rate_limit,
        daily_generation_limit=daily_limit,
        quota_database=tmp_path / "quota.sqlite3",
        quota_secret="test-quota-secret",
        client_ip_header=client_ip_header,
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


def headers(token=None, origin="http://testserver", client_ip=None):
    result = {"Origin": origin, "X-EADLD-Request": "1"}
    if token is not None:
        result["Authorization"] = f"Bearer {token}"
    if client_ip is not None:
        result["CF-Connecting-IP"] = client_ip
    return result


def test_home_has_security_headers(tmp_path):
    response = make_client(tmp_path).get("/")
    assert response.status_code == 200
    assert "default-src 'self'" in response.headers["content-security-policy"]
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["cache-control"] == "no-store"
    assert "访问码" not in response.text
    assert "生成后直接追迹，不调用优化器" not in response.text
    assert "791395970@qq.com" in response.text


def test_generation_returns_only_sanitized_metrics_and_protected_images(tmp_path):
    client = make_client(tmp_path)
    response = client.post("/api/generate", json=valid_payload(), headers=headers())
    assert response.status_code == 200
    assert response.json()["metrics"] == {
        "elements": 9,
        "efl_mm": 74.172,
        "bfl_mm": 6.731,
        "ttl_mm": 54.802,
        "mean_rms_um": 9.967,
        "worst_rms_um": 12.093,
        "valid_ray_percent": 100.0,
    }
    assert "never-return" not in response.text
    assert "prescription" not in response.text
    assert response.json()["requested_candidates"] == 3
    assert response.json()["returned_candidates"] == 2
    assert [row["rank"] for row in response.json()["candidates"]] == [1, 2]

    layout_url = response.json()["images"]["layout"]
    assert client.get(layout_url).status_code == 404
    image = client.get(
        layout_url,
        headers={"X-EADLD-Request": "1"},
    )
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/png"
    second_layout_url = response.json()["candidates"][1]["images"]["layout"]
    second_image = client.get(
        second_layout_url,
        headers={"X-EADLD-Request": "1"},
    )
    assert second_image.status_code == 200
    assert second_image.content.endswith(b"layout-2")

    seq_url = response.json()["files"]["seq"]
    assert client.get(seq_url).status_code == 404
    seq = client.get(
        seq_url,
        headers={"X-EADLD-Request": "1"},
    )
    assert seq.status_code == 200
    assert seq.headers["content-type"] == "application/octet-stream"
    assert 'attachment; filename="eadld_initial_structure.seq"' in seq.headers["content-disposition"]
    assert "TITLE 'TEST'" in seq.text


def test_rejects_origin_and_extra_fields(tmp_path):
    client = make_client(tmp_path)
    assert client.post(
        "/api/generate", json=valid_payload(), headers=headers(origin="https://attacker.invalid")
    ).status_code == 403
    payload = {**valid_payload(), "backend_factory": "attacker.module:factory"}
    assert client.post("/api/generate", json=payload, headers=headers()).status_code == 422


def test_accepts_same_origin_https_tunnel_and_rejects_spoofed_origin(tmp_path):
    client = make_client(tmp_path, allowed_hosts=("testserver", "*.trycloudflare.com"))
    public_headers = headers(origin="https://demo.trycloudflare.com") | {
        "Host": "demo.trycloudflare.com"
    }
    assert client.post(
        "/api/generate", json=valid_payload(), headers=public_headers
    ).status_code == 200
    assert client.post(
        "/api/generate",
        json=valid_payload(),
        headers=headers(origin="https://attacker.invalid")
        | {"Host": "demo.trycloudflare.com"},
    ).status_code == 403


def test_optional_access_token_still_protects_private_deployments(tmp_path):
    client = make_client(tmp_path, access_token="a" * 32)
    assert client.post("/api/generate", json=valid_payload(), headers=headers()).status_code == 401
    assert client.post(
        "/api/generate", json=valid_payload(), headers=headers(token="a" * 32)
    ).status_code == 200


def test_rate_limit_and_path_validation(tmp_path):
    client = make_client(tmp_path, rate_limit=1)
    assert client.post("/api/generate", json=valid_payload(), headers=headers()).status_code == 200
    assert client.post("/api/generate", json=valid_payload(), headers=headers()).status_code == 429
    assert client.get(
        "/api/results/not-a-job/layout",
        headers={"X-EADLD-Request": "1"},
    ).status_code == 404


def test_daily_quota_is_persistent_and_reported(tmp_path):
    client = make_client(tmp_path, daily_limit=1)
    assert client.get("/api/quota").json() == {"limit": 1, "remaining": 1}
    response = client.post("/api/generate", json=valid_payload(), headers=headers())
    assert response.status_code == 200
    assert response.json()["quota"] == {"limit": 1, "remaining": 0}

    restarted_client = make_client(tmp_path, daily_limit=1)
    assert restarted_client.get("/api/quota").json()["remaining"] == 0
    response = restarted_client.post("/api/generate", json=valid_payload(), headers=headers())
    assert response.status_code == 429
    assert response.json()["detail"] == "今日体验次数已用完"


def test_trusted_cloudflare_header_separates_visitors(tmp_path):
    client = make_client(tmp_path, daily_limit=1, client_ip_header="cf-connecting-ip")
    first = client.post(
        "/api/generate", json=valid_payload(), headers=headers(client_ip="192.0.2.10")
    )
    second = client.post(
        "/api/generate", json=valid_payload(), headers=headers(client_ip="192.0.2.11")
    )
    assert first.status_code == 200
    assert second.status_code == 200
