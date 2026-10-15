"""Derive the private reference capture configuration from kernel input and binding."""
from copy import deepcopy


class BindingError(ValueError):
    pass


def _resolve(value, kernel):
    if isinstance(value, dict) and set(value) == {'input'}:
        pointer = value['input']
        if not isinstance(pointer, str) or not pointer.startswith('/'):
            raise BindingError('input reference must be a JSON pointer starting with /')
        current = kernel
        try:
            for token in pointer[1:].split('/'):
                token = token.replace('~1', '/').replace('~0', '~')
                if isinstance(current, list):
                    if not token.isdigit() or (len(token) > 1 and token.startswith('0')):
                        raise ValueError('invalid array index')
                    current = current[int(token)]
                else:
                    current = current[token]
        except (KeyError, IndexError, ValueError, TypeError) as error:
            raise BindingError(f'input field not found: {pointer}') from error
        return deepcopy(current)
    if isinstance(value, list):
        return [_resolve(x, kernel) for x in value]
    if isinstance(value, dict):
        return {k: _resolve(v, kernel) for k, v in value.items()}
    return value


def output_bindings(binding):
    output = binding.get('outputs', {})
    tensors = output.get('tensors', []) if output.get('kind') == 'dense_tensors' else [output] if output.get('kind') == 'dense_tensor' else []
    result = []
    for tensor in tensors:
        if 'left_allocation' not in tensor or 'shape' not in tensor:
            raise BindingError('output binding requires left_allocation and shape')
        entry = dict(allocation=tensor['left_allocation'], shape=tensor['shape'])
        if tensor.get('left_offset', 0):
            entry['offset'] = tensor['left_offset']
        result.append(entry)
    return result


def derive_reference_input(kernel, binding):
    """Resolve explicit bindings; never guess dtype from byte width or scalar values."""
    try:
        reference = binding.get('reference')
        if not isinstance(reference, dict) or not isinstance(reference.get('arguments'), list):
            raise BindingError('reference.arguments is required in reference_binding.json')
        if set(reference) - {'arguments', 'outputs'}:
            raise BindingError('unknown reference binding field')
        layouts = binding.get('left_inputs', {}).get('layouts', {})
        parameters = {}
        for argument in kernel.get('arguments', []):
            parameter = argument.get('parameter')
            if parameter is None:
                continue
            parameters.setdefault((parameter, argument.get('offset', 0)), []).append(argument)
        arguments, names = [], set()
        for raw in reference['arguments']:
            arg = _resolve(raw, kernel)
            name = arg.get('name')
            if not isinstance(name, str) or not name or name in names:
                raise BindingError('reference argument names must be unique nonempty strings')
            names.add(name)
            if 'parameter' in arg:
                if set(arg) - {'name', 'parameter', 'parameter_offset'}:
                    raise BindingError(f'{name}: parameter binding cannot also specify a value or tensor')
                parameter = arg['parameter']
                candidates = parameters.get((parameter, arg.get('parameter_offset', 0)), [])
                if len(candidates) != 1 or 'value' not in candidates[0]:
                    raise BindingError(f'{name}: kernel parameter {parameter} has a missing or ambiguous concrete value')
                arg = dict(name=name, value=candidates[0]['value'])
            elif 'value' not in arg:
                allocation = arg.get('allocation')
                layout = layouts.get(allocation, {})
                if 'shape' not in arg and 'shape' in layout:
                    arg['shape'] = _resolve(layout['shape'], kernel)
                if 'offset' not in arg and layout.get('source_offset', 0):
                    arg['offset'] = layout['source_offset']
                for field in ('allocation', 'shape', 'dtype'):
                    if field not in arg:
                        raise BindingError(f'{name}: {field} cannot be derived; supply it in reference.arguments')
            allowed = {'name', 'value'} if 'value' in arg else {'name', 'allocation', 'shape', 'dtype', 'offset'}
            if set(arg) - allowed:
                raise BindingError(f'{name}: unknown or conflicting argument binding fields')
            arguments.append(arg)
        outputs = _resolve(reference['outputs'], kernel) if 'outputs' in reference else _resolve(output_bindings(binding), kernel)
        return dict(schema='spverify.pytorch-reference.v1', arguments=arguments, outputs=outputs)
    except (KeyError, TypeError, AttributeError) as error:
        raise BindingError(f'invalid reference binding: {error}') from error
