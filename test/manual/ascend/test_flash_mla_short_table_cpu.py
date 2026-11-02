"""CPU regression for FlashMLA eager metadata with short block tables.

Extracts the real method without importing NPU packages. Run with
TORCH_DEVICE_BACKEND_AUTOLOAD=0 on Ascend hosts to keep this CPU-only.
"""

import ast
import math
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

SOURCE = (
    Path(__file__).resolve().parents[3]
    / "python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py"
)


def _load_method():
    tree = ast.parse(SOURCE.read_text(), filename=str(SOURCE))
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_fmla_decode"
    )
    bucket = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_fmla_kv_bucket"
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            bucket,
            method,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {"torch": torch, "math": math}
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace


METHODS = _load_method()


class TestFlashMLAShortBlockTableCPU(unittest.TestCase):
    def _run_case(self, seq_len, table_pages, expected_max, *, graph=False, padded=0):
        batch, heads, rank, rope = 2, 3, 8, 4
        calls = {}
        q_dtype = torch.float32
        block_tables = torch.zeros((batch, table_pages), dtype=torch.int32)
        seq_lens = torch.tensor([seq_len - 1, seq_len], dtype=torch.int32)
        md = object()
        output = torch.arange(batch * heads * rank, dtype=q_dtype).reshape(
            batch, 1, heads, rank
        )
        q = torch.zeros((batch + padded, heads, rank))
        q_rope = torch.zeros((batch + padded, heads, rope))
        cache = torch.zeros((table_pages, 128, 1, rank + rope))

        def metadata(**kwargs):
            calls["metadata"] = kwargs
            return md

        def combine_query(nope, pe):
            calls["combined_shapes"] = (nope.shape, pe.shape)
            return torch.cat((nope, pe), dim=-1)

        def decode(query, key, table, lengths, **kwargs):
            calls["decode"] = kwargs
            self.assertEqual(query.shape, (batch, 1, heads, rank + rope))
            self.assertIs(key, cache)
            self.assertIs(table, block_tables)
            self.assertIs(lengths, seq_lens)
            self.assertIs(kwargs["md"], md)
            return output

        backend = SimpleNamespace(
            graph_mode=graph,
            page_size=128,
            kv_lora_rank=rank,
            qk_rope_head_dim=rope,
            forward_metadata=SimpleNamespace(
                block_tables=block_tables,
                seq_lens=seq_lens,
                fmla_seqused_kv=seq_lens,
                fmla_max_seqlen_kv=METHODS["_fmla_kv_bucket"](seq_len),
            ),
            _fmla_metadata_for=metadata,
            token_to_kv_pool=SimpleNamespace(
                get_flash_mla_kv_buffer=lambda layer: cache
            ),
        )
        adapter = SimpleNamespace(combine_query=combine_query, decode=decode)
        layer = SimpleNamespace(tp_q_head_num=heads, layer_id=5, scaling=0.125)
        with mock.patch.dict(METHODS, {"fmla_adapter": adapter}):
            with mock.patch.object(
                torch.Tensor,
                "item",
                side_effect=AssertionError("FlashMLA table capacity needs no tensor sync"),
            ):
                actual = METHODS["_fmla_decode"](
                    backend, q, q_rope, layer, None, batch + padded
                )
        self.assertEqual(calls["metadata"]["max_seqlen_kv"], expected_max)
        self.assertEqual(calls["decode"]["max_seqlen_kv"], expected_max)
        self.assertIs(calls["metadata"]["cache_seqlens"], seq_lens)
        self.assertEqual(calls["decode"]["scale"], layer.scaling)
        self.assertEqual(actual.shape, (batch + padded, heads * rank))
        self.assertTrue(torch.equal(actual[:batch], output.reshape(batch, -1)))
        if padded:
            self.assertTrue(
                torch.equal(actual[batch:], torch.zeros(padded, heads * rank))
            )
        self.assertEqual(
            calls["combined_shapes"],
            ((batch, 1, heads, rank), (batch, 1, heads, rope)),
        )

    def test_short_129_token_table_clamps_bucket_to_256(self):
        self._run_case(129, 2, 256)

    def test_page_boundary_128_clamps_one_page_to_128(self):
        self._run_case(128, 1, 128)

    def test_table_with_capacity_keeps_512_bucket(self):
        self._run_case(129, 4, 512)

    def test_long_1537_token_table_clamps_2048_bucket_to_1664(self):
        self._run_case(1537, 13, 1664)

    def test_graph_dynamic_max_minus_one_is_unchanged(self):
        self._run_case(129, 2, -1, graph=True)

    def test_padding_shape_and_zero_rows_are_preserved(self):
        self._run_case(129, 2, 256, padded=3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
