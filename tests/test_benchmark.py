from benchmark import evaluate, new_throttling

TARGETS = {"fps": 10, "frame_p95_ms": 100, "cpu_cores": 3, "rss_mb": 500}


def report(throttled):
    return {
        "fps": 10.0,
        "cpu_cores": 1.0,
        "rss_mb": 300,
        "throttled": throttled,
        "latency_ms": {"frame": {"p95": 50.0}},
    }


def test_sticky_bits_from_before_the_run_are_ignored():
    assert new_throttling(0x50000, 0x50000) is False


def test_sticky_bit_set_during_the_run_counts():
    assert new_throttling(0x0, 0x40000) is True


def test_active_throttling_counts_even_if_already_set():
    assert new_throttling(0x50005, 0x50005) is True


def test_unavailable_throttling_is_unknown():
    assert new_throttling(None, None) is None


def test_unmeasured_throttling_does_not_pass_the_gate():
    assert evaluate(report(None), TARGETS)["sem throttling"] is None
    assert evaluate(report(True), TARGETS)["sem throttling"] is False
    assert evaluate(report(False), TARGETS)["sem throttling"] is True
