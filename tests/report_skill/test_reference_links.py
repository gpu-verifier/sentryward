"""Citations must open original records, without changing quoted evidence."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    'reference_result_view', ROOT / 'skills/generate-kernel-report/scripts/result_view.py')
view = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(view)


class ReferenceLinkTests(unittest.TestCase):
    def test_nested_arrays_and_escaped_keys_use_original_lines(self):
        raw = '{\r\n "result": "unsupported",\r\n "diagnostics": [\r\n  {\r\n   "a/b~c": null\r\n  },\r\n  {"result": "nested"}\r\n ]\r\n}'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'output.json'
            path.write_bytes(raw.encode())
            self.assertEqual(view._json_pointer_lines(path, {
                '', '/result', '/diagnostics/0', '/diagnostics/0/a~1b~0c',
                '/diagnostics/1/result', '/absent', '/diagnostics/01', '/diagnostics/2',
            }), {'': 1, '/result': 2, '/diagnostics/0': 4,
                 '/diagnostics/0/a~1b~0c': 5, '/diagnostics/1/result': 7})

    def test_compact_json_and_last_duplicate_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'output.json'
            path.write_text('{"front\\u0065nd":{"old":1},\n"frontend":{"new":2}}')
            self.assertEqual(view._json_pointer_lines(path, {
                '/frontend', '/frontend/old', '/frontend/new', '/front~end',
            }), {'/frontend': 2, '/frontend/new': 2})

    def test_result_and_draft_references_link_but_quoted_data_does_not(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inp, out, ptx = [root / name for name in ('input.json', 'output.json', 'kernel.ptx')]
            inp.write_text('{\n "arguments": [\n  {"value": 2}\n ]\n}')
            out.write_text('{\n "result": "unsupported",\n "diagnostics": [\n  {"code": "stop"}\n ]\n}')
            ptx.write_text('.entry test(){}')
            document = ('<code>output.json#/result</code>\n'
                        '`output.json#/diagnostics/0`\n'
                        '`input.json#/arguments/0/value`\n'
                        '<code>output.json#/absent</code>\n'
                        '<a href="old"><code>output.json#/result</code></a>\n'
                        '<pre><code>output.json#/result</code></pre>\n'
                        '```json\n"output.json#/result"\n```\n')
            rendered = view.link_references(document, ptx.read_text(), [inp, out, ptx], root / 'docs/report.md')
            for target in ('../output.json#L2', '../output.json#L4', '../input.json#L3'):
                self.assertIn('href="' + target + '"', rendered)
            self.assertIn('<code>output.json#/absent</code>', rendered)
            self.assertIn('<a href="old"><code>output.json#/result</code></a>', rendered)
            self.assertIn('<pre><code>output.json#/result</code></pre>', rendered)
            self.assertIn('```json\n"output.json#/result"\n```', rendered)
            self.assertEqual(view.link_references(rendered, ptx.read_text(), [inp, out, ptx], root / 'docs/report.md'), rendered)

    def test_ptx_links_require_matching_file_and_physical_line(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ptx = root / 'kernel.ptx'
            ptx.write_text('.entry test(){\n ret;\n}\n')
            correct = '<code>kernel.ptx</code> · source line <code>2</code>'
            wrong = '<code>other.ptx</code> · source line <code>2</code>'
            outside = '<code>kernel.ptx</code> · source line <code>9</code>'
            caption = '<p><strong>Stop location</strong> · physical PTX 2</p>'
            document = '\n'.join((correct, wrong, outside, caption, 'instruction ID <code>2</code>'))
            rendered = view.link_references(document, ptx.read_text(), [None, None, ptx], root / 'docs/report.md')
            self.assertIn('<a href="../kernel.ptx#L2">' + correct + '</a>', rendered)
            self.assertIn('<a href="../kernel.ptx#L2">physical PTX 2</a>', rendered)
            self.assertIn(wrong, rendered)
            self.assertIn(outside, rendered)
            self.assertEqual(rendered.count('href='), 2)

    def test_moe_links_open_the_exact_pinned_record(self):
        case = ROOT / 'examples/vllm__moe_topk_softmax__sm_100a'
        paths = [case / 'input.json', case / 'output.json',
                 ROOT / 'corpus/vllm__moe_topk_softmax__sm_100a.ptx']
        inp, result = [json.loads(path.read_text()) for path in paths[:2]]
        ptx = paths[2].read_text()
        base = 'https://github.com/gpu-verifier/sentryward/blob/94a6b26'
        report = ROOT / 'docs/report-skill/moe/report.md'
        raw = view.render_result(inp, result, ptx, paths, report_path=report, artifact_base=base)
        linked = view.link_references(raw, ptx, paths, report, base)
        numbers = view._json_pointer_lines(paths[1], {'/result', '/diagnostics/0'})
        for pointer, line in numbers.items():
            url = base + '/' + paths[1].relative_to(ROOT).as_posix() + '#L' + str(line)
            self.assertIn('<a href="' + url + '"><code>output.json#' + pointer + '</code></a>', linked)
        self.assertIn('href="' + base + '/' + paths[2].relative_to(ROOT).as_posix() + '#L352"', linked)
        self.assertNotIn(base + '/examples/' + case.name + '/output.json#/', linked)

    def test_wrapped_instruction_keeps_source_link_and_wrapping_note(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ptx = root / 'kernel.ptx'
            ptx.write_text('.entry test(){\n ret;\n}\n')
            document = ('<p><strong>Second access · saved source location</strong>'
                        ' · physical PTX 2 · display wrapping added</p>\n'
                        '<pre><code>ret;</code></pre>')
            rendered = view.link_references(document, ptx.read_text(), [None, None, ptx], root / 'docs/report.md')
            self.assertIn('<a href="../kernel.ptx#L2">physical PTX 2</a> · display wrapping added</p>', rendered)
            self.assertIn('<pre><code>ret;</code></pre>', rendered)
            self.assertEqual(view.link_references(rendered, ptx.read_text(), [None, None, ptx], root / 'docs/report.md'), rendered)

    def test_github_ptx_entry_has_raw_fallback_for_large_source_files(self):
        ptx = ROOT / 'corpus/cutlass77_blackwell_fmha_fp16.compute_100a.ptx'
        report = ROOT / 'docs/report-skill/fmha/report.md'
        base = 'https://github.com/gpu-verifier/sentryward/blob/94a6b26'
        rendered = view.artifact_entry({}, [None, None, ptx], report, artifact_base=base)
        self.assertIn('[PTX](' + base + '/corpus/' + ptx.name + ')', rendered)
        self.assertIn('[raw](' + base.replace('/blob/', '/raw/') + '/corpus/' + ptx.name + ')', rendered)
        local = view.artifact_entry({}, [None, None, ptx], report)
        self.assertNotIn('[raw]', local)


if __name__ == '__main__':
    unittest.main()
