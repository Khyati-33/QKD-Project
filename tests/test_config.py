from run_experiment import resolve_config


def test_bc_randomize_endpoints_survives_config_resolution():
    resolved = resolve_config({
        "models": ["GNN"],
        "training": {"bc_randomize_endpoints": True},
    })
    assert resolved["training"]["bc_randomize_endpoints"] is True
