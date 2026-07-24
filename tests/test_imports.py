def test_public_imports() -> None:
    from m4 import S1SurfaceFieldDataset, S1SurfaceFieldPredictor

    assert S1SurfaceFieldDataset.__name__ == "S1SurfaceFieldDataset"
    assert S1SurfaceFieldPredictor.__name__ == "S1SurfaceFieldPredictor"
