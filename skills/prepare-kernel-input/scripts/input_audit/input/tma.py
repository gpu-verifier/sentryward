"""Validate optional, structured launch tensor maps."""

from .model import InputContractError, TmaDescriptorInput


ELEMENT_BYTES = {"u8": 1, "s8": 1, "e4m3": 1, "e5m2": 1,
                 "u16": 2, "s16": 2, "f16": 2, "bf16": 2,
                 "u32": 4, "s32": 4, "f32": 4, "tf32": 4,
                 "u64": 8, "s64": 8, "f64": 8}
PACKED_TYPES = {"16u4_align16b": (4, 16), "16u4_align8b": (4, 8)}
VECTOR_FIELDS = {"box_shape", "global_shape", "coordinate_strides_bytes", "element_strides"}


def parse_descriptors(records):
    from .parse import _object, _list, _integer, _string, _selector
    result = []
    for index, record in enumerate(_list(records, "$.tma_descriptors")):
        path = f"$.tma_descriptors[{index}]"
        record = _object(record, VECTOR_FIELDS | {
            "parameter", "offset", "object", "element_type", "rank", "base_byte_offset",
            "swizzle", "interleave", "oob_fill", "global_strides_bytes"}, path)
        fields = {}
        for key in ("object", "element_type", "swizzle", "interleave", "oob_fill"):
            if key in record:
                fields[key] = _string(record[key], path + "." + key)
        if "element_type" in fields and fields["element_type"] not in ELEMENT_BYTES | PACKED_TYPES:
            raise InputContractError("unsupported_tma_element_type", path + ".element_type",
                                     "unsupported tensor-map element type", status="ptx_execution_unsupported")
        for key, choices in (("swizzle", {"none", "32B", "64B", "128B", "128B_atom32B", "128B_atom64B"}),
                             ("interleave", {"none", "16B", "32B"}), ("oob_fill", {"zero", "nan"})):
            if key in fields and fields[key] not in choices:
                raise InputContractError("invalid_tma_descriptor", path + "." + key,
                                         "unknown tensor-map mode")
        for key in VECTOR_FIELDS:
            if key in record:
                fields[key] = [_integer(v, path + "." + key,
                                       minimum=0 if key == "coordinate_strides_bytes" and i else 1)
                               for i, v in enumerate(_list(record[key], path + "." + key))]
                if not 1 <= len(fields[key]) <= 5:
                    raise InputContractError("invalid_tma_descriptor", path + "." + key,
                                             "expected one to five dimensions")
        if "rank" in record:
            fields["rank"] = _integer(record["rank"], path + ".rank", minimum=1)
            if fields["rank"] > 5:
                raise InputContractError("invalid_tma_descriptor", path + ".rank", "rank exceeds five")
        ranks = {len(fields[k]) for k in VECTOR_FIELDS if k in fields}
        if "rank" in fields:
            ranks.add(fields["rank"])
        if len(ranks) > 1:
            raise InputContractError("inconsistent_tma_descriptor", path, "shape and stride ranks disagree")
        if "global_strides_bytes" in record:
            if "coordinate_strides_bytes" in record:
                raise InputContractError("inconsistent_tma_descriptor", path, "choose one stride representation")
            fields["global_strides_bytes"] = [_integer(v, path + ".global_strides_bytes", minimum=0)
                for v in _list(record["global_strides_bytes"], path + ".global_strides_bytes")]
            if ranks and len(fields["global_strides_bytes"]) + 1 != next(iter(ranks)):
                raise InputContractError("inconsistent_tma_descriptor", path, "outer stride count must be rank minus one")
        if fields.get("element_type") in PACKED_TYPES and "coordinate_strides_bytes" in fields:
            raise InputContractError("inconsistent_tma_descriptor", path,
                                     "packed types require outer global_strides_bytes")
        if "base_byte_offset" in record:
            fields["base_byte_offset"] = _integer(record["base_byte_offset"], path + ".base_byte_offset")
            if fields["base_byte_offset"] >= 1 << 64:
                raise InputContractError("invalid_tma_descriptor", path, "base offset exceeds u64")
        if (fields.get("interleave") == "none" and "coordinate_strides_bytes" in fields
                and fields.get("element_type") in ELEMENT_BYTES
                and fields["coordinate_strides_bytes"][0] != ELEMENT_BYTES[fields["element_type"]]):
            raise InputContractError("inconsistent_tma_descriptor", path,
                                     "non-interleaved dimension zero must have the element byte stride")
        result.append(TmaDescriptorInput(_selector(record.get("parameter"), path + ".parameter"),
            fields, _integer(record.get("offset", 0), path + ".offset")))
    return tuple(result)
