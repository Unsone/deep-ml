def balance_undersample(data: list) -> list:
    """
    Undersample the majority classes so all classes have the same number of
    samples equal to the minority class count.

    data: list of (sample, label) tuples
    Returns: list of (sample, label) tuples, order-preserving
    """
    if not data:
        return []

    counts = {}
    for _, label in data:
        counts[label] = counts.get(label, 0) + 1

    min_count = min(counts.values())

    kept = {}
    result = []
    for sample, label in data:
        if kept.get(label, 0) < min_count:
            result.append((sample, label))
            kept[label] = kept.get(label, 0) + 1

    return result