"""Bounded exact rewrites of portable expression artifacts."""
from .graph import Graph

def pair_fold_limit(left,right,requested):
    """Avoid expanding only one side of a larger reduction."""
    if requested < 1 or requested > 64:
        raise ValueError("fold unroll limit must be between 1 and 64")
    counts=[n["fold"]["count"] for a in (left,right) for n in a["nodes"] if "fold" in n]
    return min(requested,4) if any(count>requested for count in counts) else requested


def small_folds(artifact, limit=4):
    graph = Graph()
    cache = {}
    def visit(ref, removed):
        key = (ref["node"], removed)
        if key in cache:
            result = dict(cache[key])
        else:
            node = artifact["nodes"][ref["node"]]
            bindings = dict(removed)
            level = lambda x: x-sum(k < x for k in bindings)
            if node["kind"] == "Index":
                old = node["index_level"]
                result = graph.constant(bindings[old]) if old in bindings else graph.index(level(old))
            elif node["kind"] == "Fold" and node["fold"]["count"] <= limit:
                fold = node["fold"]
                result = visit(node["operands"][0], removed)
                operation = {"Sum": "Add", "Product": "Multiply", "Max": "Maximum", "Min": "Minimum"}[fold["reducer"]]
                for k in range(fold["count"]):
                    body = visit(node["operands"][1], tuple(sorted((*removed, (fold["binder_level"], k)))))
                    result = graph.op(operation, result, body, dtype=node["type"])
            else:
                args = [visit(x, removed) for x in node["operands"]]
                extra = {}
                if "fold" in node:
                    extra["fold"] = {**node["fold"], "binder_level": level(node["fold"]["binder_level"])}
                result = graph.node(node["kind"], node["type"], node["name"], args, attributes=node["attributes"], **extra)
            cache[key] = dict(result)
        for name in ("conversion", "converted_type", "pointer_provenance"):
            if name in ref:
                result[name] = ref[name]
        return result
    families = [{**f, "body": visit(f["body"], ())} for f in artifact["output_families"]]
    return graph.artifact(families)
