"""Pure-CPU ordering regression for the real expanded KV getter.

Guarded lists reject reading K/V before their layer-transfer wait. No torch,
NPU, or optional framework packages are imported by this standalone test.
"""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

SOURCE = (
    Path(__file__).resolve().parents[3]
    / "python/sglang/srt/hardware_backend/npu/memory_pool_npu.py"
)


def load_getter():
    tree = ast.parse(SOURCE.read_text(), filename=str(SOURCE))
    pool = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "NPUMLATokenToKVPool"
    )
    getter = next(
        node for node in pool.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_prefill_expanded_kv"
    )
    namespace = {}
    exec(compile(ast.Module(body=[getter], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["get_prefill_expanded_kv"]


GETTER = load_getter()


class GuardedList:
    def __init__(self, name, values, events, waiting):
        self.name, self.values, self.events, self.waiting = name, values, events, waiting

    def __getitem__(self, index):
        if self.waiting is not None and index not in self.waiting:
            raise AssertionError(f"{self.name}[{index}] read before layer wait")
        self.events.append((self.name, index))
        return self.values[index]


def pool_fixture(start_layer, *, with_counter=True, wait_error=None):
    events = []
    waited = set() if with_counter else None
    keys, values = [object() for _ in range(3)], [object() for _ in range(3)]

    def wait(index):
        events.append(("wait", index))
        if wait_error is not None:
            raise wait_error
        waited.add(index)

    pool = SimpleNamespace(
        start_layer=start_layer,
        layer_transfer_counter=SimpleNamespace(wait_until=wait) if with_counter else None,
        prefill_k_buffer=GuardedList("key", keys, events, waited),
        prefill_v_buffer=GuardedList("value", values, events, waited),
    )
    return pool, events, keys, values


class TestHiCacheExpandedFenceCPU(unittest.TestCase):
    def test_wait_precedes_both_buffer_reads(self):
        pool, events, keys, values = pool_fixture(5)
        actual = GETTER(pool, 7)
        self.assertIs(actual[0], keys[2])
        self.assertIs(actual[1], values[2])
        self.assertEqual(events, [("wait", 2), ("key", 2), ("value", 2)])

    def test_zero_start_first_layer_waits_local_zero(self):
        pool, events, keys, values = pool_fixture(0)
        actual = GETTER(pool, 0)
        self.assertIs(actual[0], keys[0])
        self.assertIs(actual[1], values[0])
        self.assertEqual(events, [("wait", 0), ("key", 0), ("value", 0)])

    def test_no_counter_preserves_direct_getter(self):
        pool, events, keys, values = pool_fixture(10, with_counter=False)
        actual = GETTER(pool, 11)
        self.assertIs(actual[0], keys[1])
        self.assertIs(actual[1], values[1])
        self.assertEqual(events, [("key", 1), ("value", 1)])

    def test_each_getter_call_waits_for_its_own_local_layer(self):
        pool, events, _, _ = pool_fixture(5)
        GETTER(pool, 6)
        GETTER(pool, 5)
        GETTER(pool, 6)
        self.assertEqual(events, [
            ("wait", 1), ("key", 1), ("value", 1),
            ("wait", 0), ("key", 0), ("value", 0),
            ("wait", 1), ("key", 1), ("value", 1),
        ])

    def test_failed_wait_does_not_access_either_buffer(self):
        error = RuntimeError("transfer not ready")
        pool, events, _, _ = pool_fixture(5, wait_error=error)
        with self.assertRaisesRegex(RuntimeError, "not ready"):
            GETTER(pool, 7)
        self.assertEqual(events, [("wait", 2)])


if __name__ == "__main__":
    unittest.main(verbosity=2)
