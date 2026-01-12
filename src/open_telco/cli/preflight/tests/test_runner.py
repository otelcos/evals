"""Tests for PreflightRunner.run_model_test()."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest

from open_telco.cli.preflight.runner import (
    PreflightConfig,
    PreflightRunner,
    PreflightStatus,
)

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.parametrize(
    "eval_return,expected",
    [
        ((True, {}), True),
        ((False, {}), False),
    ],
)
@pytest.mark.asyncio
async def test_run_model_test_returns_expected(
    eval_return: tuple[bool, dict[str, Any]],
    expected: bool,
    default_config: PreflightConfig,
) -> None:
    mock_func = MagicMock(return_value=eval_return)
    runner = PreflightRunner(eval_func=mock_func, config=default_config)
    result = await runner.run_model_test("model")
    assert result is expected


@pytest.mark.asyncio
async def test_run_model_test_timeout_returns_none(
    mock_slow_eval_func: Callable[..., tuple[bool, Any]],
    short_timeout_config: PreflightConfig,
) -> None:
    runner = PreflightRunner(eval_func=mock_slow_eval_func, config=short_timeout_config)
    result = await runner.run_model_test("model")
    assert result is None


@pytest.mark.asyncio
async def test_run_model_test_cancelled_returns_none(
    mock_eval_func: MagicMock,
    default_config: PreflightConfig,
) -> None:
    runner = PreflightRunner(eval_func=mock_eval_func, config=default_config)
    runner.cancel()
    result = await runner.run_model_test("model")
    assert result is None


@pytest.mark.asyncio
async def test_run_model_test_exception_propagates(
    mock_raising_eval_func: MagicMock,
    default_config: PreflightConfig,
) -> None:
    runner = PreflightRunner(eval_func=mock_raising_eval_func, config=default_config)
    with pytest.raises(RuntimeError):
        await runner.run_model_test("model")


@pytest.mark.asyncio
async def test_run_model_test_cancelled_does_not_call_eval(
    mock_eval_func: MagicMock,
    default_config: PreflightConfig,
) -> None:
    runner = PreflightRunner(eval_func=mock_eval_func, config=default_config)
    runner.cancel()
    await runner.run_model_test("model")
    assert mock_eval_func.called is False


@pytest.mark.parametrize(
    "eval_return,expected_status",
    [
        ((True, {}), PreflightStatus.PASSED),
        ((False, {}), PreflightStatus.FAILED),
    ],
)
@pytest.mark.asyncio
async def test_progress_callback_final_status(
    eval_return: tuple[bool, dict[str, Any]],
    expected_status: PreflightStatus,
    default_config: PreflightConfig,
    progress_tracker: dict[str, list[tuple[str, PreflightStatus]]],
    progress_callback: Callable[[str, PreflightStatus], None],
) -> None:
    mock_func = MagicMock(return_value=eval_return)
    runner = PreflightRunner(
        eval_func=mock_func,
        config=default_config,
        on_progress=progress_callback,
    )
    await runner.run_model_test("model")
    assert progress_tracker["calls"][-1][1] is expected_status


@pytest.mark.asyncio
async def test_progress_callback_running_status_first(
    mock_eval_func: MagicMock,
    default_config: PreflightConfig,
    progress_tracker: dict[str, list[tuple[str, PreflightStatus]]],
    progress_callback: Callable[[str, PreflightStatus], None],
) -> None:
    runner = PreflightRunner(
        eval_func=mock_eval_func,
        config=default_config,
        on_progress=progress_callback,
    )
    await runner.run_model_test("model")
    assert progress_tracker["calls"][0][1] is PreflightStatus.RUNNING


@pytest.mark.asyncio
async def test_progress_callback_timeout_status(
    mock_slow_eval_func: Callable[..., tuple[bool, Any]],
    short_timeout_config: PreflightConfig,
    progress_tracker: dict[str, list[tuple[str, PreflightStatus]]],
    progress_callback: Callable[[str, PreflightStatus], None],
) -> None:
    runner = PreflightRunner(
        eval_func=mock_slow_eval_func,
        config=short_timeout_config,
        on_progress=progress_callback,
    )
    await runner.run_model_test("model")
    assert progress_tracker["calls"][-1][1] is PreflightStatus.TIMEOUT


@pytest.mark.asyncio
async def test_progress_callback_skipped_status(
    mock_eval_func: MagicMock,
    default_config: PreflightConfig,
    progress_tracker: dict[str, list[tuple[str, PreflightStatus]]],
    progress_callback: Callable[[str, PreflightStatus], None],
) -> None:
    runner = PreflightRunner(
        eval_func=mock_eval_func,
        config=default_config,
        on_progress=progress_callback,
    )
    runner.cancel()
    await runner.run_model_test("model")
    assert progress_tracker["calls"][-1][1] is PreflightStatus.SKIPPED


@pytest.mark.asyncio
async def test_no_progress_callback_does_not_raise(
    mock_eval_func: MagicMock,
    default_config: PreflightConfig,
) -> None:
    runner = PreflightRunner(
        eval_func=mock_eval_func,
        config=default_config,
        on_progress=None,
    )
    result = await runner.run_model_test("model")
    assert result is True


@pytest.mark.parametrize(
    "param_name,expected_value_fn",
    [
        ("tasks", lambda cfg: list(cfg.tasks)),
        ("model", lambda _: ["test-model"]),
        ("limit", lambda cfg: cfg.sample_limit),
        ("log_dir", lambda cfg: cfg.log_dir),
    ],
)
@pytest.mark.asyncio
async def test_eval_func_receives_param(
    param_name: str,
    expected_value_fn: Callable[[PreflightConfig], Any],
    mock_eval_func: MagicMock,
    default_config: PreflightConfig,
) -> None:
    runner = PreflightRunner(eval_func=mock_eval_func, config=default_config)
    await runner.run_model_test("test-model")
    assert mock_eval_func.call_args.kwargs[param_name] == expected_value_fn(default_config)


@pytest.mark.parametrize(
    "results,expected_count",
    [
        ({"m1": True, "m2": True}, 2),
        ({"m1": True, "m2": False}, 1),
        ({"m1": False, "m2": None}, 0),
        ({}, 0),
    ],
)
def test_get_passed_models_count(
    results: dict[str, bool | None],
    expected_count: int,
    mock_eval_func: MagicMock,
    default_config: PreflightConfig,
) -> None:
    runner = PreflightRunner(eval_func=mock_eval_func, config=default_config)
    runner.results = results
    assert len(runner.get_passed_models()) == expected_count


@pytest.mark.parametrize(
    "results,expected_count",
    [
        ({"m1": False, "m2": False}, 2),
        ({"m1": True, "m2": False}, 1),
        ({"m1": True, "m2": None}, 0),
        ({}, 0),
    ],
)
def test_get_failed_models_count(
    results: dict[str, bool | None],
    expected_count: int,
    mock_eval_func: MagicMock,
    default_config: PreflightConfig,
) -> None:
    runner = PreflightRunner(eval_func=mock_eval_func, config=default_config)
    runner.results = results
    assert len(runner.get_failed_models()) == expected_count


@pytest.mark.asyncio
async def test_run_all_returns_all_results(
    mock_eval_func: MagicMock,
    default_config: PreflightConfig,
) -> None:
    runner = PreflightRunner(eval_func=mock_eval_func, config=default_config)
    results = await runner.run_all(["m1", "m2", "m3"])
    assert len(results) == 3


@pytest.mark.asyncio
async def test_run_all_stores_results(
    mock_eval_func: MagicMock,
    default_config: PreflightConfig,
) -> None:
    runner = PreflightRunner(eval_func=mock_eval_func, config=default_config)
    await runner.run_all(["m1", "m2"])
    assert len(runner.results) == 2


@pytest.mark.parametrize(
    "results,expected_summary",
    [
        ({"m1": True, "m2": True}, "2/2 passed"),
        ({"m1": True, "m2": False}, "1/2 passed"),
        ({"m1": False, "m2": None}, "0/2 passed"),
    ],
)
def test_get_summary_passed_count(
    results: dict[str, bool | None],
    expected_summary: str,
    mock_eval_func: MagicMock,
    default_config: PreflightConfig,
) -> None:
    runner = PreflightRunner(eval_func=mock_eval_func, config=default_config)
    runner.results = results
    assert expected_summary in runner.get_summary()
