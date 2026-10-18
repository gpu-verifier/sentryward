"""Bounded PTX parameter and guard inspection; not verifier findings or reachability proof."""

import html
import json
import re

MISSING = object()


def parameter_bindings(run_input, ptx):
    """Index parameter loads in the selected entry; no branch execution inferred."""
    entry = run_input.get("entry")
    if not isinstance(entry, str):
        return []
    match = re.search(r"\.entry\s+" + re.escape(entry) + r"(?=\s|\()", ptx)
    if not match:
        return []
    opening = ptx.find("{", match.end())
    if opening == -1:
        return []
    depth = 0
    end = None
    for index in range(opening, len(ptx)):
        if ptx[index] == "{":
            depth += 1
        elif ptx[index] == "}":
            depth -= 1
            if depth == 0:
                end = index + 1
                break
    if end is None:
        raise ValueError("Selected PTX entry has an unbalanced body.")
    declarations = {
        (str(d.get("parameter")), d.get("offset", 0)): d.get("name")
        for d in run_input.get("declarations", [])
        if isinstance(d, dict)
    }
    aliases = {}
    result = []
    for number, line in enumerate(ptx[:end].splitlines(), 1):
        if number <= ptx[:opening].count("\n"):
            continue
        code = line.split("//", 1)[0]
        assignment = re.search(r"\b[\w.]+\s+(%\w+)\s*,", code)
        alias = re.search(
            r"\bmov\.\w+\s+(%\w+),\s*" + re.escape(entry) + r"_param_(\d+)\s*;", code
        )
        if assignment:
            aliases.pop(assignment.group(1), None)
        if alias:
            aliases[alias.group(1)] = alias.group(2)
        load = re.search(r"\bld\.param\.\w+\s+(%\w+),\s*\[([^\]]+)\]", code)
        if not load:
            continue
        address = load.group(2).replace(" ", "")
        direct = re.fullmatch(re.escape(entry) + r"_param_(\d+)(?:\+(\d+))?", address)
        indirect = re.fullmatch(r"(%\w+)(?:\+(\d+))?", address)
        parameter = (
            direct.group(1)
            if direct
            else aliases.get(indirect.group(1))
            if indirect
            else None
        )
        offset = int((direct or indirect).group(2) or 0) if direct or indirect else None
        name = declarations.get((parameter, offset))
        if name is not None:
            result.append(
                {
                    "field": name,
                    "register": load.group(1),
                    "line": number,
                    "offset": offset,
                }
            )
    return result


def constant_predicates(run_input, ptx, bindings):
    """Fold integer-scalar predicates with unique definitions; no CFG inference."""
    entry = run_input.get("entry", "")
    start = re.search(r"\.entry\s+" + re.escape(entry) + r"(?=\s|\()", ptx)
    if not start:
        return []
    next_entry = re.search(r"\.(?:entry|func)\b", ptx[start.end() :])
    end = start.end() + next_entry.start() if next_entry else len(ptx)
    lines = ptx[:end].splitlines()
    first_line = ptx[: start.start()].count("\n") + 1
    definitions = {}
    for number, line in enumerate(lines, 1):
        if number < first_line:
            continue
        code = line.split("//", 1)[0].strip()
        match = re.match(r"(?:@!?%\w+\s+)?([\w.]+)\s+(%\w+),\s*(.*?)\s*;", code)
        if match:
            definitions.setdefault(match.group(2), []).append(
                (number, match.group(1), match.group(3))
            )
    names = {
        (str(d.get("parameter")), d.get("offset", 0)): d.get("name")
        for d in run_input.get("declarations", [])
        if isinstance(d, dict)
    }
    arguments = {}
    for argument in run_input.get("arguments", []):
        if not isinstance(argument, dict) or "value" not in argument:
            continue
        name = argument.get("field") or names.get(
            (str(argument.get("parameter")), argument.get("offset", 0))
        )
        if name is not None:
            arguments[name] = argument["value"]
    types = {
        d.get("name"): d.get("type")
        for d in run_input.get("declarations", [])
        if isinstance(d, dict)
    }
    known = {}
    named_inputs = {}
    for binding in bindings:
        register = binding["register"]
        value = arguments.get(binding["field"])
        width = re.fullmatch(r"[us](\d+)", str(types.get(binding["field"])))
        load = definitions.get(register, [])
        load_width = (
            re.fullmatch(r"ld\.param\.[bus](\d+)", load[0][1])
            if len(load) == 1
            else None
        )
        integer = (
            width
            and load_width
            and width.group(1) == load_width.group(1)
            and type(value) is int
            and 0 <= value < 2 ** int(width.group(1))
        )
        null_pointer = (
            types.get(binding["field"]) == "pointer"
            and binding["field"] in arguments
            and value is None
            and load_width
            and load_width.group(1) in ("32", "64")
        )
        if integer or null_pointer:
            known[register] = 0 if null_pointer else value
            named_inputs[register] = binding["field"]
    proofs = []

    def value(token):
        token = token.strip()
        if token in known:
            return known[token]
        try:
            return int(token, 0)
        except ValueError:
            return None

    for register, instructions in sorted(
        definitions.items(), key=lambda item: item[1][0][0]
    ):
        if len(instructions) != 1:
            continue
        number, opcode, raw = instructions[0]
        operands = [part.strip() for part in raw.split(",")]
        values = [value(part) for part in operands]
        folded = None
        if (
            re.fullmatch(r"setp\.(?:eq|ne)\.(?:b|s|u)\d+", opcode)
            and len(values) == 2
            and all(v is not None and v >= 0 for v in values)
        ):
            folded = (
                values[0] == values[1] if ".eq." in opcode else values[0] != values[1]
            )
        elif opcode == "or.pred" and len(values) == 2:
            folded = (
                True
                if any(v is True for v in values)
                else False
                if all(v is False for v in values)
                else None
            )
        elif opcode == "and.pred" and len(values) == 2:
            folded = (
                False
                if any(v is False for v in values)
                else True
                if all(v is True for v in values)
                else None
            )
        elif opcode == "not.pred" and len(values) == 1 and type(values[0]) is bool:
            folded = not values[0]
        if folded is not None:
            known[register] = folded
            proofs.append(
                {
                    "predicate": register,
                    "value": folded,
                    "line": number,
                    "expression": opcode + " " + raw,
                    "inputs": {
                        named_inputs[token]: known[token]
                        for token in operands
                        if token in named_inputs
                    },
                }
            )
    return proofs


def predicate_table(proofs):
    if not proofs:
        return ""
    body = "Unique-definition scalar and null-pointer simplifications. Values apply **when these definitions are reached**; they do not prove path reachability. A true OR operand fixes the result even if the other operand is unknown.\n\n"
    body += "| Predicate | Value / physical PTX | Definition |\n| --- | --- | --- |\n"
    for proof in proofs:
        definition = key_cell(proof["expression"])
        if proof.get("inputs"):
            definition += "<br/>" + ", ".join(
                key_cell(field) + " = " + cell(value)
                for field, value in proof["inputs"].items()
            )
        body += (
            "| "
            + key_cell(proof["predicate"])
            + " | "
            + cell(proof["value"])
            + " / "
            + str(proof["line"])
            + " | "
            + definition
            + " |\n"
        )
    return body


def skipped_parameter_loads(run_input, ptx, bindings, proofs):
    """Forward known-taken branches, excluding outside entries into their gap."""
    entry = run_input.get("entry")
    if not isinstance(entry, str):
        return []
    match = re.search(r"\.entry\s+" + re.escape(entry) + r"(?=\s|\()", ptx)
    if not match:
        return []
    opening = ptx.find("{", match.end())
    if opening == -1:
        return []
    depth = 0
    closing = None
    for index in range(opening, len(ptx)):
        if ptx[index] == "{":
            depth += 1
        elif ptx[index] == "}":
            depth -= 1
            if depth == 0:
                closing = index
                break
    if closing is None:
        return []
    lines = ptx.splitlines()
    first = ptx[:opening].count("\n") + 1
    last = ptx[:closing].count("\n") + 1
    body = [
        (n, line.split("//", 1)[0])
        for n, line in enumerate(lines, 1)
        if first <= n <= last
    ]
    if any(re.search(r"\bbrx\.", line) for _, line in body):
        return []
    values = {p["predicate"]: (p["value"], p["line"]) for p in proofs}
    # Include every direct branch in the selected entry, including branches
    # after the final parameter load that could re-enter a skipped gap.
    labels = {
        m.group(1): n
        for n, line in body
        if (m := re.match(r"\s*([A-Za-z_$][\w$]*):", line))
    }
    branches = []
    for n, line in body:
        match = re.match(
            r"\s*(?:@(!?)(%\w+)\s+)?bra(?:\.uni)?\s+([A-Za-z_$][\w$]*)\s*;", line
        )
        if match:
            branches.append((n, match.group(1), match.group(2), match.group(3)))
    skipped = []
    for n, negative, predicate, target in branches:
        if predicate not in values or target not in labels:
            continue
        value, definition = values[predicate]
        if definition >= n:
            continue
        taken = not value if negative else value
        end = labels[target]
        if not taken or end <= n:
            continue
        inner_labels = {label for label, where in labels.items() if n < where < end}
        if any(
            (where <= n or where >= end) and destination in inner_labels
            for where, _, _, destination in branches
        ):
            continue
        for binding in bindings:
            if n < binding["line"] < end:
                proof = next(p for p in proofs if p["predicate"] == predicate)
                skipped.append(
                    {
                        **binding,
                        "branch": n,
                        "target": end,
                        "predicate": predicate,
                        "condition": proof["expression"],
                        "inputs": proof.get("inputs", {}),
                    }
                )
    return skipped


def skipped_table(items):
    if not items:
        return ""
    body = "Forward branches fixed by supplied integer values bypass these parameter loads. No outside direct branch enters the skipped gap. This is static source support, not a saved runtime trace.\n\n"
    body += "| Load / field | Known-taken branch | Target |\n| --- | --- | --- |\n"
    for item in items:
        guard = key_cell(item["predicate"])
        if item.get("inputs"):
            guard += "<br/>" + ", ".join(
                key_cell(field) + " = " + cell(value)
                for field, value in item["inputs"].items()
            )
        body += (
            f"| {item['line']} / "
            + key_cell(item["field"])
            + f" | {item['branch']} / "
            + guard
            + f" | {item['target']} |\n"
        )
    return body


def instruction_notes(ptx):
    if not re.search(r"^\s*\.version\s+9\.2\b", ptx, re.M):
        return ""
    if not ("tcgen05.cp." in ptx and re.search(r"tcgen05\.mma(?:\.sp)?\.", ptx)):
        return ""
    return (
        "**PTX 9.2 source-check boundary.** `tcgen05.cp.cta_group::N` → `tcgen05.mma.cta_group::N` is a pipelined pairing for the same N. "
        "The same-thread pattern needs no explicit ordering mechanism. Before attributing a candidate to a missing wait, check the issuing-thread relation and applicability of this pairing. "
        "A saved effect byte count is not the full instruction transfer; do not convert TMEM register offsets into saved byte offsets without a supplied mapping.\n\n"
        "[Pipelined pairings](https://docs.nvidia.com/cuda/archive/13.2.0/parallel-thread-execution/index.html#tcgen05-memory-consistency-model-pipelined-instructions) · "
        "[Same-thread pattern](https://docs.nvidia.com/cuda/archive/13.2.0/parallel-thread-execution/index.html#tcgen05-memory-consistency-model-canonical-sync-patterns-pipelined-same-thread)\n"
    )


def binding_table(bindings):
    body = "Parameter loads in the selected PTX entry. **Listing a load does not prove its branch is reached.** Pointer and stride are different fields. For a packed load, the name identifies the field at its starting offset, not the entire loaded register; inspect the instruction width.\n\n"
    body += "| Start field / byte offset | Destination | Physical PTX |\n| --- | --- | --- |\n"
    for binding in bindings:
        body += (
            "| "
            + key_cell(binding["field"])
            + " / "
            + str(binding["offset"])
            + " | "
            + key_cell(binding["register"])
            + " | "
            + str(binding["line"])
            + " |\n"
        )
    return body


def control_index(run_input, result, ptx):
    """List source branches and definitions; do not infer their reachability."""
    bindings = parameter_bindings(run_input, ptx)
    if not bindings:
        return ""
    lines = ptx.splitlines()
    first = min(b["line"] for b in bindings)
    end_match = re.search(r"\.(?:entry|func)\b", "\n".join(lines[first:]))
    last = (
        first + "\n".join(lines[first:])[: end_match.start()].count("\n")
        if end_match
        else len(lines)
    )
    code = {
        n: line.split("//", 1)[0].strip()
        for n, line in enumerate(lines, 1)
        if first <= n <= last
    }
    labels = {
        m.group(1): n
        for n, line in code.items()
        if (m := re.fullmatch(r"([A-Za-z_$][\w$]*):", line))
    }
    branches = [
        (n, m.group(1), m.group(2))
        for n, line in code.items()
        if (
            m := re.match(
                r"(?:@(!?%\w+)\s+)?bra(?:\.uni)?\s+([A-Za-z_$][\w$]*)\s*;", line
            )
        )
    ]
    definitions = {}
    for n, line in code.items():
        match = re.match(r"(?:@!?%\w+\s+)?[\w.]+\s+(%\w+)(?:\|%\w+)?,", line)
        if match:
            definitions.setdefault(match.group(1), []).append((n, line))

    def support(predicate, before):
        if predicate is None:
            return "Unconditional branch; taken whenever its block is reached. Block reachability is separate."
        pending = [predicate.lstrip("!")]
        seen = set()
        found = []
        for _ in range(4):
            next_pending = []
            for register in pending:
                if register in seen:
                    continue
                seen.add(register)
                candidates = [
                    (n, text) for n, text in definitions.get(register, []) if n < before
                ]
                if candidates:
                    n, text = candidates[-1]
                    found.append(f"{n}: " + key_cell(text))
                    next_pending.extend(re.findall(r"%\w+", text)[1:])
            pending = next_pending
        return "<br/>".join(found) or "no indexed definition"

    body = "Exact direct-branch index. Listed definitions are nearest earlier definitions, not a reaching-definition proof. Check the configured path separately.\n\n"
    proofs = constant_predicates(run_input, ptx, bindings)
    waits = [
        (n, line)
        for n, line in code.items()
        if re.search(r"cp\.async\.bulk\.wait_group\.read\s+0\s*;", line)
    ]
    for n, line in waits:
        previous = [
            (ln, text)
            for ln, text in code.items()
            if ln < n and text and text not in ("{", "}")
        ]
        if not previous or not previous[-1][1].endswith(":"):
            continue
        label = previous[-1][1][:-1]
        body += f"**Final read wait {n}, block " + key_cell(label) + ".**\n\n"
        entrances = [
            branch for branch, _, target in branches if target == label and branch < n
        ]
        if entrances:
            body += branch_corridor(code, labels, proofs, min(entrances), n) + "\n"
        body += (
            "| Entrance | Exact source | Earlier definitions |\n| --- | --- | --- |\n"
        )
        for branch, predicate, target in branches:
            if target == label:
                body += (
                    f"| Direct jump / {branch} | "
                    + key_cell(code[branch])
                    + " | "
                    + support(predicate, branch)
                    + " |\n"
                )
        predecessor = previous[:-1]
        if predecessor:
            at, text = predecessor[-1]
            predicate = next((p for b, p, _ in branches if b == at), None)
            evidence = (
                support(predicate, at)
                if any(b == at for b, _, _ in branches)
                else "Not a direct branch; inspect the preceding block."
            )
            body += (
                f"| Fallthrough predecessor / {at} | "
                + key_cell(text)
                + " | "
                + evidence
                + " |\n"
            )
    locations = [
        record.get(side, {}).get("location", {}).get("line")
        for kind in ("races", "warnings")
        for record in result.get(kind, [])
        if isinstance(record, dict)
        for side in ("first", "second")
    ]
    locations = [n for n in locations if type(n) is int]
    return_labels = set()
    for label, number in labels.items():
        following = [
            (n, text)
            for n, text in code.items()
            if n > number and text and text not in ("{", "}")
        ]
        if following and re.match(r"ret\s*;", following[0][1]):
            return_labels.add(label)
    exits = [
        (n, p, t)
        for n, p, t in branches
        if locations and n < min(locations) and t in return_labels
    ]
    if exits:
        body += "\n**Earlier branches to return before the recorded source location.** These may narrow local eligibility; barrier counts alone do not identify participating threads.\n\n"
        body += "| Physical PTX | Exact branch | Earlier definitions |\n| --- | --- | --- |\n"
        for n, predicate, target in exits[-3:]:
            body += (
                f"| {n} | " + key_cell(code[n]) + " | " + support(predicate, n) + " |\n"
            )
    return body


def branch_corridor(code, labels, proofs, start, end):
    """Follow a bounded local corridor using only indexed constant predicates.

    Entry to start and the indexed definitions' reachability are assumptions,
    not proved here. Stop at an unknown decision rather than guessing it.
    """
    known = {p["predicate"]: (p["value"], p["line"]) for p in proofs}
    branches = []
    definitions = {}
    seen = set()
    at = start
    while start <= at <= end and at not in seen:
        seen.add(at)
        line = code.get(at, "")
        definition = re.match(r"(?:@!?%\w+\s+)?[\w.]+\s+(%\w+)(?:\|%\w+)?,", line)
        if definition:
            definitions[definition.group(1)] = (at, line)
        branch = re.match(
            r"(?:@(!?)(%\w+)\s+)?bra(?:\.uni)?\s+([A-Za-z_$][\w$]*)\s*;", line
        )
        if branch:
            negative, predicate, target = branch.groups()
            folded = known.get(predicate)
            if predicate is not None and (folded is None or folded[1] >= at):
                origin = definitions.get(predicate)
                detail = (
                    (
                        "Latest definition **on this corridor**: "
                        + str(origin[0])
                        + ": "
                        + key_cell(origin[1])
                    )
                    if origin
                    else "Value not evaluated by this bounded index."
                )
                branches.append(
                    f"| {at} | "
                    + key_cell(line)
                    + " | **Unknown; stop here.** "
                    + detail
                    + " |"
                )
                break
            taken = (
                True
                if predicate is None
                else (not folded[0] if negative else folded[0])
            )
            branches.append(
                f"| {at} | "
                + key_cell(line)
                + " | "
                + ("Taken" if taken else "Not taken")
                + " |"
            )
            if taken:
                target_line = labels.get(target)
                if target_line is None or not start <= target_line <= end:
                    break
                at = target_line
                continue
        at += 1
    if not branches:
        return ""
    return (
        "**Local branch corridor.** Assuming entry to "
        + str(start)
        + " and the indexed constant definitions, this path follows branches until an unknown decision. It is not a full kernel trace. Definitions in bypassed blocks are not used.\n\n"
        "| Physical PTX | Branch | Decision / source |\n| --- | --- | --- |\n"
        + "\n".join(branches)
        + "\n"
    )


def mma_flag_index(run_input, ptx):
    """Index adjacent MMA flag comparisons and all their source definitions.

    This is listing evidence, not control-flow or loop reaching-value analysis.
    """
    from code_excerpt import mask_noncode

    clean = mask_noncode(ptx)
    entry = run_input.get("entry")
    if not isinstance(entry, str):
        return ""
    start = re.search(r"\.entry\s+" + re.escape(entry) + r"(?=\s|\()", clean)
    if not start:
        return ""
    opening = clean.find("{", start.end())
    if opening < 0:
        return ""
    depth, closing = 0, None
    for i in range(opening, len(clean)):
        if clean[i] == "{":
            depth += 1
        if clean[i] == "}":
            depth -= 1
            if depth == 0:
                closing = i
                break
    if closing is None:
        raise ValueError("Selected PTX entry has an unbalanced body.")
    first = clean[:opening].count("\n") + 1
    last = clean[:closing].count("\n") + 1
    lines = clean.splitlines()
    originals = ptx.splitlines()
    definitions = {}
    for n in range(first, last + 1):
        line = lines[n - 1].strip()
        m = re.match(r"(?:@!?[\w%]+\s+)?[\w.]+\s+(%\w+)\s*,", line)
        if m:
            definitions.setdefault(m[1], []).append((n, originals[n - 1].strip()))
    rows = []
    for n in range(first + 1, last + 1):
        if not re.search(r"\btcgen05\.mma\.", lines[n - 1]):
            continue
        flag = re.search(
            r"setp\.ne\.b32\s+([\w%]+)\s*,\s*(%\w+|\d+)\s*,\s*0\s*;", lines[n - 2]
        )
        if not flag or not re.search(
            r",\s*" + re.escape(flag[1]) + r"\s*;", lines[n - 1]
        ):
            continue
        operand = flag[2]
        support = definitions.get(operand, [])
        shown = (
            "<br/>".join(str(at) + ": " + key_cell(text) for at, text in support)
            if support
            else key_cell(operand) + " (literal)"
            if operand.isdigit()
            else "No indexed definition; inspect source."
        )
        rows.append(f"| {n} | {n - 1} / " + key_cell(operand) + " | " + shown + " |\n")
    if not rows:
        return ""
    return (
        "\n**MMA accumulate-flag source index.** Comparisons are adjacent to the MMA in the listing. "
        "All indexed assignments to their input are shown, including later loop updates. "
        "**Do not treat the nearest earlier assignment as a loop reaching-value proof.** "
        "Trace initialization versus subsequent iterations.\n\n"
        "| MMA / physical PTX | Flag comparison / input | Input assignments in this entry |\n| --- | --- | --- |\n"
        + "".join(rows)
    )


def local_copy_guards(run_input, ptx):
    """Evaluate simple local copy guards, never full-path reachability.

    Only unique, unpredicated mov/and/setp definitions over tid.x and unsigned
    32-bit literals are understood. Unknown or reused definitions stay unknown.
    """
    entry = run_input.get("entry")
    if not isinstance(entry, str):
        return []
    clean = re.sub(
        r"/\*.*?\*/", lambda m: "\n" * m.group().count("\n"), ptx, flags=re.S
    )
    start = re.search(r"\.entry\s+" + re.escape(entry) + r"(?=\s|\()", clean)
    if not start:
        return []
    opening = clean.find("{", start.end())
    if opening < 0:
        return []
    depth, closing = 0, None
    for index in range(opening, len(clean)):
        depth += (clean[index] == "{") - (clean[index] == "}")
        if depth == 0:
            closing = index
            break
    if closing is None:
        return []
    first = clean[:opening].count("\n") + 1
    last = clean[:closing].count("\n") + 1
    body = [
        (n, line.split("//", 1)[0].strip())
        for n, line in enumerate(clean.splitlines(), 1)
        if first <= n <= last
    ]
    definitions, labels = {}, {}
    for n, code in body:
        label = re.match(r"([A-Za-z_$][\w$]*):", code)
        if label:
            labels[label.group(1)] = n
        for definition in re.finditer(
            r"(?:^|(?<=[;{}]))\s*(@!?%\w+\s+)?([\w.]+)\s+(%\w+)(?:\|%\w+)?,\s*([^;]+);",
            code,
        ):
            predicated, opcode, dest, args = definition.groups()
            definitions.setdefault(dest, []).append((n, bool(predicated), opcode, args))

    def expression(token, before, seen=frozenset()):
        if token == "%tid.x":
            return ("tid.x",), []
        try:
            number = int(token, 0)
            return (("literal", number), []) if 0 <= number < 2**32 else (None, [])
        except ValueError:
            pass
        found = definitions.get(token, [])
        if token in seen or len(found) != 1:
            return None, []
        n, predicated, op, args = found[0]
        if predicated or n >= before:
            return None, []
        operands = [v.strip() for v in args.split(",")]
        allowed = {
            "mov.u32": 1,
            "mov.b32": 1,
            "mov.s32": 1,
            "and.b32": 2,
            "setp.eq.b32": 2,
            "setp.ne.b32": 2,
            "setp.eq.u32": 2,
            "setp.ne.u32": 2,
            "setp.eq.s32": 2,
            "setp.ne.s32": 2,
        }
        if op not in allowed or len(operands) != allowed[op]:
            return None, []
        children, support = [], []
        for operand in operands:
            node, proof = expression(operand, n, seen | {token})
            if node is None:
                return None, []
            children.append(node)
            support += proof
        support += [(n, token, op + " " + args)]
        return (op, *children), support

    def evaluate(node, tid):
        op = node[0]
        if op == "literal":
            return node[1]
        if op == "tid.x":
            return tid
        values = [evaluate(child, tid) for child in node[1:]]
        if op.startswith("mov."):
            return values[0]
        if op == "and.b32":
            return values[0] & values[1]
        return values[0] == values[1] if ".eq." in op else values[0] != values[1]

    def describe(node):
        op = node[0]
        if op == "literal":
            return str(node[1])
        if op == "tid.x":
            return "tid.x"
        if op.startswith("mov."):
            return describe(node[1])
        symbol = "&" if op == "and.b32" else "==" if ".eq." in op else "!="
        return "(" + describe(node[1]) + " " + symbol + " " + describe(node[2]) + ")"

    block = run_input.get("launch", {}).get("block_dim", [])
    limit = (
        block[0]
        if isinstance(block, list)
        and block
        and type(block[0]) is int
        and 0 < block[0] <= 1024
        else None
    )
    answer = []
    previous_branch = None
    for n, code in body:
        if re.match(r"[A-Za-z_$][\w$]*:", code):
            previous_branch = None
        branch = re.fullmatch(
            r"(?:@(!?)(%\w+)\s+)?bra(?:\.uni)?\s+([A-Za-z_$][\w$]*)\s*;", code
        )
        if branch:
            previous_branch = (n, *branch.groups())
            continue
        if not re.match(
            r"cp\.async\.bulk\.(?:tensor\.\dd\.global\.shared|wait_group\.read)", code
        ):
            continue
        if not previous_branch:
            continue
        at, negated, predicate, target = previous_branch
        if predicate is None or labels.get(target, -1) <= n:
            continue
        node, support = expression(predicate, at)
        # A branch operand must resolve to a predicate, not an integer mask.
        if node is not None and not node[0].startswith("setp."):
            node = None
        admitted = None
        if node is not None and limit is not None:
            admitted = [
                tid
                for tid in range(limit)
                if not (not evaluate(node, tid) if negated else evaluate(node, tid))
            ]
        answer.append(
            {
                "site": n,
                "operation": code.split()[0],
                "branch": at,
                "predicate": predicate,
                "negated": bool(negated),
                "skip_target": labels[target],
                "skip_condition": ("NOT " if negated else "") + describe(node)
                if node is not None
                else None,
                "accepted_tid_x": admitted,
                "support": support,
            }
        )
    return answer


def local_guard_table(items):
    if not items:
        return ""
    text = "**Isolated copy/wait guards — outer exits NOT applied.** These are literal branch sites and bounded predicate evaluations, assuming their indexed definitions are reached; dominance is not proved. They are not recorded issuers or a complete path. `tid.x` values are x coordinates; do not count them as full thread identities. Combine any required outer condition before stating a narrowed range.\n\n"
    for item in items:
        text += (
            f"- Operation {item['site']}; skip branch {item['branch']}: "
            + key_cell(("!" if item["negated"] else "") + item["predicate"])
            + f" → {item['skip_target']}. "
        )
        text += (
            "Condition: " + key_cell(item["skip_condition"] or "not evaluated") + ".\n"
        )
        values = item["accepted_tid_x"]
        if values is not None:
            runs = []
            for value in values:
                if runs and value == runs[-1][1] + 1:
                    runs[-1][1] = value
                else:
                    runs.append([value, value])
            ranges = (
                ", ".join(str(a) if a == b else f"{a}–{b}" for a, b in runs) or "none"
            )
            text += f"  Passing this guard alone: **{ranges}** ({len(values)} distinct x values).\n"
        for line, register, expression in item["support"]:
            text += f"  {line}: " + key_cell(register + " ← " + expression) + "\n"
    return text


def encoded(value):
    if value is MISSING:
        return "not supplied"
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "))


def cell(value):
    # HTML code preserves exact data, escapes source markup and table pipes.
    return "<code>" + html.escape(encoded(value)).replace("|", "&#124;") + "</code>"


def key_cell(value):
    return "<code>" + html.escape(str(value)).replace("|", "&#124;") + "</code>"


def render(run_input, result, ptx):
    """Collect the source-only inspection views used by --bindings."""
    bindings = parameter_bindings(run_input, ptx)
    proofs = constant_predicates(run_input, ptx, bindings)
    return "\n".join(
        [
            "SOURCE ANALYSIS AIDS: local deductions, not verifier findings or reachability proofs.",
            binding_table(bindings),
            predicate_table(proofs),
            skipped_table(skipped_parameter_loads(run_input, ptx, bindings, proofs)),
            control_index(run_input, result, ptx),
            local_guard_table(local_copy_guards(run_input, ptx)),
            mma_flag_index(run_input, ptx),
            instruction_notes(ptx),
        ]
    )
