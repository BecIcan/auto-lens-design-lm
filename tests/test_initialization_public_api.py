from eadld.initialization import DesignSpec, LensSeed, run_generation_audit


class PublicTestBackend:
    def generate(self, spec):
        return [
            LensSeed(
                candidate_id="public-test",
                lens_sequence="s-aRa-aRa-aRa-",
                spacings_mm=(1.56274209619, 5.46374401399, 10.665431266, 11.975087064, 39.3610263407, 15.0024144163, 54.9176367768),
                curvatures_per_mm=(0.000625821571703, -0.0379287868671, -0.00106417392786, 0.0543700762226, 0.0226606901903, -0.00529313208104),
                refractive_indices_d=(1.620411, 1.717362, 1.620411),
                abbe_numbers=(60.292614, 29.517426, 60.292614),
                asphere_coefficients=(
                    (0.0, -6.36036744916e-05, 2.12294745603e-07, -5.38363660686e-11, -2.77368651596e-13),
                    (0.0, -1.67251318477e-05, 1.11181187259e-07, 1.73816745142e-10, -5.01420185265e-13),
                    (0.0, 6.14570829518e-05, -1.0845611976e-07, -1.09900156959e-10, 2.12537519473e-13),
                    (0.0, 2.54496705935e-05, -3.62035953094e-07, 1.71209881533e-10, 5.65862570868e-13),
                    (0.0, -7.02502841652e-06, -1.06712019709e-08, -1.19151990785e-10, 2.14482049152e-13),
                    (0.0, -3.13195689906e-06, -4.32667647315e-08, 5.1575749036e-11, -9.851430484e-16),
                ),
            )
        ]

    def public_metadata(self):
        return {"name": "public-test", "weights_in_repository": False}


def test_private_backend_public_physics_contract(tmp_path):
    output = tmp_path / "audit"
    manifest = run_generation_audit(
        DesignSpec(100.0, 2.8, 5.0, (486.1, 550.0, 656.3), 3, 1),
        PublicTestBackend(),
        output,
    )
    assert manifest["runtime_contract"]["optimizer_invocations"] == 0
    assert manifest["runtime_contract"]["paraxial_solves"] == 0
    assert manifest["candidates"][0]["passed"]
    assert "curvatures_per_mm" not in (output / "manifest.json").read_text(encoding="utf-8")
    assert (output / "layout.png").stat().st_size > 10_000
    assert (output / "spots.png").stat().st_size > 10_000
