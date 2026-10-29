"""The KV allocation plus its dummy page must leave the transient budget free."""
from types import SimpleNamespace

import pytest

from sglang.srt.model_executor import pool_configurator as sizing


def test_reserve_validation(monkeypatch):
    kvc = SimpleNamespace(device='npu')
    monkeypatch.setattr(sizing, 'get_disagg', lambda: SimpleNamespace(disaggregation_mode='prefill'))
    monkeypatch.setenv('SGLANG_NPU_PREFILL_TRANSIENT_RESERVE_MB', '0')
    assert sizing._npu_prefill_transient_reserve_bytes(SimpleNamespace(device='cpu')) == 0
    monkeypatch.setenv('SGLANG_NPU_PREFILL_TRANSIENT_RESERVE_MB', '3')
    assert sizing._npu_prefill_transient_reserve_bytes(kvc) == 3 * 1024**2
    with pytest.raises(ValueError, match='NPU PD prefill'):
        sizing._npu_prefill_transient_reserve_bytes(SimpleNamespace(device='cuda'))
    monkeypatch.setattr(sizing, 'get_disagg', lambda: SimpleNamespace(disaggregation_mode='decode'))
    with pytest.raises(ValueError, match='NPU PD prefill'):
        sizing._npu_prefill_transient_reserve_bytes(kvc)
    monkeypatch.setenv('SGLANG_NPU_PREFILL_TRANSIENT_RESERVE_MB', '-1')
    with pytest.raises(ValueError, match='nonnegative'):
        sizing._npu_prefill_transient_reserve_bytes(kvc)


@pytest.mark.parametrize('reserve_mib', [0, 3])
def test_budget_includes_reserve_and_dummy_page(reserve_mib):
    planner = sizing.DefaultPoolConfigurator.__new__(sizing.DefaultPoolConfigurator)
    planner._cell_size = 1024
    planner._expanded_prefill_kv = True
    planner._transient_prefill_reserve_bytes = reserve_mib * 1024**2
    budget, page_size = 12 * 1024**2, 128
    config = planner.calculate_pool_sizes(budget, page_size)
    allocated = (config.max_total_num_tokens + page_size) * planner._cell_size
    free = budget - allocated
    assert planner._transient_prefill_reserve_bytes <= free
    assert free - planner._transient_prefill_reserve_bytes < page_size * planner._cell_size
    assert config.max_total_num_tokens % page_size == 0
    # An explicit token cap is an absolute count, not a second byte budget.
    assert planner.calculate_pool_sizes_from_max_tokens(1000, page_size).max_total_num_tokens == 896


def test_oversized_reserve_rejected():
    planner = sizing.DefaultPoolConfigurator.__new__(sizing.DefaultPoolConfigurator)
    planner._cell_size = 1024
    planner._expanded_prefill_kv = True
    planner._transient_prefill_reserve_bytes = 12 * 1024**2
    with pytest.raises(RuntimeError, match='less than one KV page'):
        planner.calculate_pool_sizes(12 * 1024**2, 128)
