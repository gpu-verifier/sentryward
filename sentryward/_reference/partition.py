"""Exact compact output-coordinate partitioning for dense tensor mappings."""
import copy
import json
import math

from .graph import Graph, strides
from .lowering import Unsupported


def clone(source, graph, root, coordinates, old_rank, new_rank=1, *, contexts=None):
    # A partition changes the requested root, not necessarily its coordinate
    # substitution. Reuse mapped ancestors only under the exact same source,
    # destination and binder substitution. The cache is local to prepare(),
    # whose source artifacts are immutable and alive throughout this call.
    key = (id(source), id(graph), old_rank, new_rank,
           tuple(json.dumps(c, sort_keys=True) for c in coordinates))
    mapped = {} if contexts is None else contexts.setdefault(key, {})
    def use(ref):
        result = dict(mapped[ref["node"]])
        for key in ("conversion", "converted_type", "pointer_provenance"):
            if key in ref:
                result[key] = ref[key]
        return result
    stack = [(root["node"], False)]
    while stack:
        i, ready = stack.pop()
        if i in mapped:
            continue
        n = source["nodes"][i]
        if not ready:
            stack.append((i, True))
            stack.extend((r["node"], False) for r in n["operands"] if r["node"] not in mapped)
            continue
        if n["kind"] == "Index":
            level = n["index_level"]
            result = coordinates[level] if level < old_rank else graph.index(level-old_rank+new_rank)
        else:
            extra = {}
            if "fold" in n:
                extra["fold"] = {**n["fold"], "binder_level": n["fold"]["binder_level"]-old_rank+new_rank}
            result = graph.node(n["kind"], n["type"], n["name"], [use(x) for x in n["operands"]], attributes=n["attributes"], **extra)
        mapped[i] = result
    return use(root)


def _rectangles(left, right, mapping, axes, families):
    """Prove coverage in coordinate space; never enumerate output elements.

    F families and D tensor dimensions cost O(F D^2 + F^2 D) for axis
    decomposition and the disjointness certificate, with O(F D) metadata.
    Cloning visits each family's reachable DAG once per binding
    context. A family axis may fuse physically contiguous mapped axes with
    exact mixed-radix alignment. No output elements are enumerated.
    """
    m = mapping["outputs"]
    width, shape = m["element_bytes"], m["shape"]
    boxes = []
    for f in families:
        if (f["allocation"] != m["right_allocation"] or f["element_bytes"] != width or
            not f["shape"] or len(f["shape"]) != len(f["strides_bytes"]) or
            any(n <= 0 for n in f["shape"])):
            raise Unsupported("unsupported rectangular output family")
        remaining = f["offset"] - m["right_offset"]
        if remaining < 0:
            raise Unsupported("optimized output precedes mapped tensor")
        base = [0] * len(shape)
        for stride, size, axis in reversed(axes):
            base[axis], remaining = divmod(remaining, stride)
            if base[axis] >= size:
                raise Unsupported("optimized output writes outside mapped tensor")
        if remaining:
            raise Unsupported("optimized output writes into padding or is unaligned")
        counts, bindings = [1] * len(shape), {}
        for dim, (size, stride) in enumerate(zip(f["shape"], f["strides_bytes"])):
            if size == 1:
                continue
            matches = [position for position, (step, _, _) in enumerate(axes)
                       if stride > 0 and stride % step == 0]
            if not matches:
                raise Unsupported("output family axis is not a mapped tensor slice")
            # Several family axes can tile ONE tensor coordinate, e.g.
            # column = 8*register_group + payload_component. Preserve both
            # binders; prove their sum is a dense, injective mixed radix below.
            position, divisor, remaining_size = matches[-1], 1, size
            slope = stride // axes[position][0]
            while True:
                step, extent, axis = axes[position]
                if (remaining_size-1)*slope < extent-base[axis]:
                    bindings.setdefault(axis, []).append((dim, divisor, None, slope, remaining_size))
                    break
                # Carrying into another axis is rectangular only after a full,
                # aligned inner radix. A partial first/last row is not silently
                # widened to a rectangle; padding cannot be traversed either.
                if slope != 1 or base[axis] or remaining_size % extent:
                    raise Unsupported("output family fused axis is not radix aligned")
                if position+1 == len(axes) or axes[position+1][0] != step*extent:
                    raise Unsupported("output family fused axis crosses padding or tensor extent")
                bindings.setdefault(axis, []).append((dim, divisor, extent, 1, extent))
                remaining_size //= extent
                divisor *= extent
                position += 1
        for axis, terms in bindings.items():
            count = 1
            for _, _, _, slope, size in sorted(terms, key=lambda t: t[3]):
                if slope != count:
                    raise Unsupported("output family axes overlap or leave holes in mapped coordinates")
                count *= size
            if base[axis] + count > shape[axis]:
                raise Unsupported("output family axes exceed mapped coordinate extent")
            counts[axis] = count
        for previous, extents, _ in boxes:
            if all(x < p+n and p < x+k for x, k, p, n in zip(base, counts, previous, extents)):
                raise Unsupported("optimized output families overlap")
        boxes.append((base, counts, bindings))
    if sum(math.prod(counts) for _, counts, _ in boxes) != math.prod(shape):
        raise Unsupported("optimized output coverage differs from full reference output")

    a, b = Graph(), Graph()
    contexts = {}
    fa, fb, maps = [], [], []
    for f, (base, _, bindings) in zip(families, boxes):
        rank = len(f["shape"])
        coords = []
        for axis, offset in enumerate(base):
            coordinate = a.constant(offset)
            for dim, divisor, radix, slope, _ in bindings.get(axis, []):
                index = a.index(dim)
                if divisor != 1:
                    index = a.op("Divide", index, a.constant(divisor))
                if radix is not None:
                    index = a.op("Remainder", index, a.constant(radix))
                if slope != 1:
                    index = a.op("Multiply", index, a.constant(slope))
                coordinate = a.op("Add", coordinate, index)
            coords.append(coordinate)
        av = clone(left, a, left["output_families"][0]["body"], coords, len(shape), rank, contexts=contexts)
        bv = clone(right, b, f["body"], [b.index(i) for i in range(rank)], rank, rank, contexts=contexts)
        name, steps = f"output-piece-{len(fa)}", strides(f["shape"], width)
        common = dict(allocation=name, offset=0, element_bytes=width,
                      shape=f["shape"], strides_bytes=steps)
        fa.append(dict(common, body=av)); fb.append(dict(common, body=bv))
        maps.append(dict(kind="dense_tensor", left_allocation=name, right_allocation=name,
                         left_offset=0, right_offset=0, element_bytes=width, shape=f["shape"],
                         left_strides_bytes=steps, right_strides_bytes=steps))
    result = copy.deepcopy(mapping)
    result["outputs"] = {"kind": "dense_tensors", "tensors": maps}
    return a.artifact(fa), b.artifact(fb), result


def _strided_coverage(families, mapping, width, extent):
    """Certify 1-D packed ranges without expanding their logical members.

    Sweep bounding intervals, testing candidate AP intersections by CRT.
    Cost is O(P log P + C log S): P stored packed-component ranges, C
    overlapping bounding-range pairs, S maximum stride; C can be quadratic.
    """
    if extent != math.prod(mapping["shape"])*width:
        raise Unsupported("strided packed output requires a dense physical tensor")
    ranges, total = [], 0
    for f in families:
        packed, step, count = f["element_bytes"], f["strides_bytes"][0], f["shape"][0]
        begin = f["offset"]-mapping["right_offset"]
        if (f["allocation"] != mapping["right_allocation"] or packed <= 0 or
            packed % width or count <= 0 or step < packed or step % width or
            begin < 0 or begin % width or begin+(count-1)*step+packed > extent):
            raise Unsupported("invalid strided packed output range")
        total += count*packed
        # Packed scalar components, not thread/output members.
        for slot in range(packed//width):
            start, stride = begin//width+slot, step//width
            ranges.append((start, start+(count-1)*stride, stride))
    active = []
    for start, end, step in sorted(ranges):
        active = [r for r in active if r[1] >= start]
        for previous, last, stride in active:
            divisor = math.gcd(step, stride)
            delta = start-previous
            if delta % divisor:
                continue
            modulus = step//divisor
            index = 0 if modulus == 1 else (delta//divisor * pow(stride//divisor, -1, modulus)) % modulus
            first = previous+stride*index
            period = stride*modulus
            first += max(0, (start-first+period-1)//period)*period
            if first <= min(end, last):
                raise Unsupported("optimized output families overlap")
        active.append((start,end,step))
    if total != extent:
        raise Unsupported("optimized output coverage differs from the full reference output")


def prepare(left, right, mapping):
    m = mapping["outputs"]
    if m["kind"] != "dense_tensor" or left["outputs"] or right["outputs"]:
        raise Unsupported("compact partition requires dense tensor families")
    if len(left["output_families"]) != 1:
        raise Unsupported("reference must have one logical output family")
    family = left["output_families"][0]
    width = m["element_bytes"]
    if (width <= 0 or not m["shape"] or any(x <= 0 for x in m["shape"]) or
        len(m["shape"]) != len(m["right_strides_bytes"])):
        raise Unsupported("invalid dense output mapping")
    if (family["shape"] != m["shape"] or family["strides_bytes"] != m["left_strides_bytes"]
        or family["allocation"] != m["left_allocation"] or family["offset"] != m["left_offset"]
        or family["element_bytes"] != width):
        raise Unsupported("reference output does not match the declared logical tensor")
    extent = width
    axes = sorted((s, n, axis) for axis, (s, n) in enumerate(zip(m["right_strides_bytes"], m["shape"])) if n > 1)
    for stride, size, _ in axes:
        if stride < extent or stride % width:
            raise Unsupported("optimized output mapping overlaps or is unaligned")
        extent += stride * (size-1)
    families = sorted(right["output_families"], key=lambda f: f["offset"])
    if any(len(f["shape"]) != 1 or len(f["strides_bytes"]) != 1 for f in families):
        return _rectangles(left, right, mapping, axes, families)
    strided = any(f["strides_bytes"] != [f["element_bytes"]] for f in families)
    if strided and all(f["element_bytes"] == width for f in families):
        return _rectangles(left,right,mapping,axes,families)
    if strided:
        _strided_coverage(families,m,width,extent)
    cursor = m["right_offset"]
    for f in families:
        if (f["allocation"] != m["right_allocation"] or len(f["shape"]) != 1 or
            f["shape"][0] <= 0 or f["element_bytes"] <= 0 or
            (not strided and f["strides_bytes"] != [f["element_bytes"]])):
            raise Unsupported("unsupported optimized output allocation or family layout")
        if not strided and f["offset"] < cursor:
            raise Unsupported("optimized output families overlap or precede the mapped tensor")
        cursor = f["offset"] + f["shape"][0]*f["element_bytes"]
    if sum(f["shape"][0]*f["element_bytes"] for f in families) != math.prod(m["shape"])*width:
        raise Unsupported("optimized output coverage differs from the full reference output")
    a, b = Graph(), Graph()
    contexts = {}
    fa, fb, maps = [], [], []
    for f in families:
        packed_width = f["element_bytes"]
        if packed_width % width:
            raise Unsupported("output element width mismatch")
        for slot in range(packed_width//width):
            start = 0
            while start < f["shape"][0]:
                step = f["strides_bytes"][0]
                base = f["offset"]-m["right_offset"]+start*step+slot*width
                remaining = base
                initial_coordinates = [0] * len(m["shape"])
                remainders = [0] * len(m["shape"])
                for stride, size, axis in reversed(axes):
                    initial_coordinates[axis], remaining = divmod(remaining, stride)
                    remainders[axis] = remaining
                    if initial_coordinates[axis] >= size:
                        raise Unsupported("optimized output writes outside the mapped tensor")
                if remaining:
                    raise Unsupported("optimized output writes into padding or is unaligned")
                count = f["shape"][0]-start
                formulas = [(0, 0)] * len(m["shape"])
                delta = step
                # Decode the address increment in the SAME descending-stride
                # order as the base address. Once an outer coordinate consumes
                # it, inner coordinates stay fixed rather than falsely moving
                # by packed_width again (which split every row into singletons).
                for stride, size, axis in reversed(axes):
                    initial, residue = initial_coordinates[axis], remainders[axis]
                    if delta % stride == 0:
                        slope = delta//stride
                        if slope:
                            count = min(count, (size-1-initial)//slope+1)
                        formulas[axis] = (initial, slope)
                        delta = 0
                    else:
                        count = min(count, (stride-1-residue)//delta+1)
                        formulas[axis] = (initial, 0)
                coords = [a.op("Add", a.constant(c), a.op("Multiply", a.index(0), a.constant(s)))
                          for c, s in formulas]
                av = clone(left, a, family["body"], coords, len(m["shape"]), contexts=contexts)
                bi = b.op("Add", b.constant(start), b.index(0))
                bv = clone(right, b, f["body"], [bi], 1, contexts=contexts)
                if packed_width != width:
                    bits, small = 8*packed_width, 8*width
                    ty = bv.get("converted_type", b.nodes[bv["node"]]["type"])
                    if ty not in {f"b{bits}", f"u{bits}"}:
                        raise Unsupported("packed output requires an unsigned bit representation")
                    if ty != f"u{bits}":
                        bv["conversion"] = "; ".join(filter(None, [bv.get("conversion"), f"Bitcast({ty} -> u{bits})"]))
                        bv["converted_type"] = f"u{bits}"
                    if slot:
                        bv = b.op("ShiftRight", bv, b.constant(slot*small, f"u{bits}"), dtype=f"u{bits}")
                    output_type = family["body"].get("converted_type", left["nodes"][family["body"]["node"]]["type"])
                    widths = {"e4m3": 8, "e5m2": 8, "f16": 16, "bf16": 16, "f32": 32, "f64": 64}
                    widths.update({f"{kind}{n}": n for kind in "bus" for n in (8, 16, 32, 64)})
                    if widths.get(output_type) != small:
                        raise Unsupported("packed output target must have a matching scalar width")
                    steps = [bv.get("conversion"), f"Convert(u{bits} -> u{small})"]
                    if output_type != f"u{small}":
                        steps.append(f"Bitcast(u{small} -> {output_type})")
                    bv["conversion"] = "; ".join(filter(None, steps))
                    bv["converted_type"] = output_type
                name = f"output-piece-{len(fa)}"
                common = dict(allocation=name, offset=0, element_bytes=width, shape=[count], strides_bytes=[width])
                fa.append(dict(common, body=av)); fb.append(dict(common, body=bv))
                maps.append(dict(kind="dense_tensor", left_allocation=name, right_allocation=name,
                                 left_offset=0, right_offset=0, element_bytes=width, shape=[count],
                                 left_strides_bytes=[width], right_strides_bytes=[width]))
                start += count
    result = copy.deepcopy(mapping)
    result["outputs"] = {"kind": "dense_tensors", "tensors": maps}
    return a.artifact(fa), b.artifact(fb), result
