import numpy as np
from scipy.special import softmax
from sklearn.isotonic import IsotonicRegression


METHODS = ["topotransfer", "global_envelope", "feature_only", "topology_only", "source_only"]


def representation(graph):
    x, A = np.asarray(graph["x"], float), np.asarray(graph["adjacency"], float)
    degree = A.sum(1)
    normalized = A / np.maximum(degree[:, None], 1)
    return np.column_stack([x, normalized @ x])


def signature(graph, method):
    A = np.asarray(graph["adjacency"], float)
    degrees = A.sum(1)
    normalized = A / np.sqrt(np.maximum(degrees[:, None] * degrees[None], 1))
    spectrum = np.linalg.eigvalsh(normalized)
    topology = np.r_[
        np.histogram(degrees / max(len(A) - 1, 1), bins=np.linspace(0, 1, 6), density=False)[0]
        / len(A),
        spectrum[-3:],
        np.trace(A @ A @ A) / max(1, len(A) ** 3),
    ]
    embeddings = representation(graph)
    feature = np.r_[embeddings.mean(0), embeddings.std(0)]
    if method == "feature_only":
        return feature
    if method == "topology_only":
        return topology
    return np.r_[topology, feature]


def threshold(scores, alpha):
    index = int(np.ceil((len(scores) + 1) * (1 - alpha))) - 1
    return float(np.sort(scores)[index]) if index < len(scores) else 1.0


def sets(probabilities, cutoff):
    order = np.argsort(-probabilities, axis=-1, kind="stable")
    sorted_p = np.take_along_axis(probabilities, order, axis=1)
    included = np.cumsum(sorted_p, axis=1) - sorted_p < cutoff + 1e-12
    output = np.zeros_like(included)
    np.put_along_axis(output, order, included, axis=1)
    return output


def graph(rng, n, shift):
    labels = np.array([rng.randrange(2) for _ in range(n)])
    x = np.array([[float(label) + rng.gauss(0, 0.7) + shift, rng.gauss(0, 1)] for label in labels])
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            A[i, j] = A[j, i] = rng.random() < (
                0.5 - 0.3 * shift if labels[i] == labels[j] else 0.05 + 0.1 * shift
            )
    return {"x": x.tolist(), "adjacency": A.tolist(), "y": labels.tolist()}


def generate(rng, count):
    for index in range(count):
        calibration = [graph(rng, 30, rng.uniform(0, 1)) for _ in range(100)]
        yield {
            "id": str(index),
            "family": f"shift-family-{index}",
            "source": graph(rng, 60, 0),
            "calibration_environments": calibration,
            "target": graph(rng, 30, rng.uniform(0, 1.3)),
            "alpha": 0.1,
            "delta": 0.1,
            "tolerance": 0.3,
            "stability_tolerance": 0.1,
        }


def run_case(case, method):
    source = case["source"]
    source_features = representation(source)
    source_labels = np.asarray(source["y"])
    split = len(source_labels) // 2
    train_x, train_y = source_features[:split], source_labels[:split]
    W = np.linalg.solve(
        train_x.T @ train_x + np.eye(train_x.shape[1]), train_x.T @ np.eye(2)[train_y]
    )
    probability = softmax(source_features[split:] @ W * 3, axis=-1)
    labels = source_labels[split:]
    order = np.argsort(-probability, axis=1)
    cumulative = np.cumsum(np.take_along_axis(probability, order, axis=1), axis=1)
    scores = np.array(
        [cumulative[i, np.flatnonzero(order[i] == y)[0]] for i, y in enumerate(labels)]
    )
    cutoff = threshold(scores, case["alpha"])
    environments = case["calibration_environments"]
    phi0 = signature(source, method)
    phi = np.array([signature(e, method) for e in environments])
    scale = np.maximum(phi.std(0), 1e-5)
    discrepancies = np.linalg.norm((phi - phi0) / scale, axis=1)
    risks = []
    for e in environments:
        prediction = sets(softmax(representation(e) @ W * 3, axis=-1), cutoff)
        risks.append(np.mean(~prediction[np.arange(len(e["y"])), e["y"]]))
    risks = np.asarray(risks)
    middle = len(environments) // 2
    model = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(
        discrepancies[:middle], risks[:middle] - case["alpha"]
    )
    residual = risks[middle:] - case["alpha"] - model.predict(discrepancies[middle:])
    q = threshold(residual, case["delta"])
    sizes = [len(e["y"]) for e in environments[middle:]]
    finite = np.sqrt(np.log(2 / case["delta"]) / (2 * min(sizes)))
    target_phi = signature(case["target"], method)
    d = np.linalg.norm((target_phi - phi0) / scale)
    upper = min(1.0, case["alpha"] + float(model.predict([d])[0]) + q + finite)
    supported = bool(
        np.all(target_phi >= phi.min(0))
        and np.all(target_phi <= phi.max(0))
        and discrepancies.min() <= d <= discrepancies.max()
    )
    leave_one = []
    for i in range(len(residual)):
        smaller = np.delete(residual, i)
        leave_one.append(threshold(smaller, case["delta"]))
    stable = max(abs(value - q) for value in leave_one) <= case["stability_tolerance"]
    accepted = upper <= case["tolerance"] and supported and stable
    if method == "global_envelope":
        accepted = upper <= case["tolerance"]
    if method == "source_only":
        accepted, upper = True, case["alpha"]
    target_sets = sets(softmax(representation(case["target"]) @ W * 3, axis=-1), cutoff)
    return {
        "accepted": accepted,
        "upper_bound": upper,
        "sets": target_sets.tolist(),
        "supported": supported,
        "stable": stable,
    }


def score(case, result):
    selected = np.asarray(result["sets"])
    labels = np.asarray(case["target"]["y"])
    miscoverage = float(np.mean(~selected[np.arange(len(labels)), labels]))
    return {
        "acceptance": int(result["accepted"]),
        "false_certificate": int(miscoverage > case["tolerance"]) if result["accepted"] else None,
        "target_miscoverage": miscoverage,
        "set_size": float(selected.sum(1).mean()),
        "upper_envelope": result["upper_bound"],
        "envelope_failure": int(miscoverage > result["upper_bound"]),
        "support_refusal": int(not result["supported"]),
        "stability_refusal": int(not result["stable"]),
        "legal_effects": int(result["accepted"] and miscoverage <= case["tolerance"]),
    }
