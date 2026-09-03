"""SSHVmEnvironment: real-size query + fallback (no live VM needed).

The environment's ``__init__`` opens a real SSH connection, so these tests
build the object via ``__new__`` and stub ``_run`` directly — exercising
``screen_size`` (the coordinate-drift fix) without touching the network.
"""

from minicua.desktop.ssh_vm import SSHVmEnvironment


def _make_env(run_fn, hint=(1920, 1080)):
    env = SSHVmEnvironment.__new__(SSHVmEnvironment)
    env._screen_size = hint
    env._real_screen_size = None
    env._run = run_fn
    return env


def test_screen_size_queries_guest_and_caches():
    calls = []

    def run(code):
        calls.append(code)
        return "730 624\n"

    env = _make_env(run)
    assert env.screen_size() == (730, 624)
    # The real size is cached: a second call must not re-query the guest.
    assert env.screen_size() == (730, 624)
    assert len(calls) == 1


def test_screen_size_falls_back_to_hint_when_query_fails():
    def run(code):
        return "some error output\n"

    env = _make_env(run)
    assert env.screen_size() == (1920, 1080)


def test_screen_size_falls_back_to_hint_on_empty_output():
    def run(code):
        return ""

    env = _make_env(run)
    assert env.screen_size() == (1920, 1080)


def test_screen_size_falls_back_to_hint_on_non_numeric():
    def run(code):
        return "730 abc\n"

    env = _make_env(run)
    assert env.screen_size() == (1920, 1080)
