def corrupt_features(data, features, combo, mask):
    from pipelines.seed_gnn.utils import corrupt_features as seed_corrupt_features
    return seed_corrupt_features(data, features, combo, mask)


