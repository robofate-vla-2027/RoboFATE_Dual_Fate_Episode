"""Original optional-tool configuration."""
SETTING = 'web_search="disabled"'

def disabled_search_argv(command):
    assert isinstance(command, list) and command[:2] == ["codex", "exec"]
    assert all(isinstance(arg, str) for arg in command)
    assert len(command) > 3 and command[-1].startswith("Read instruction.md.")
    options = command[2:-1]
    assert "--search" not in options
    configs = [options[i + 1] for i, arg in enumerate(options[:-1]) if arg == "-c"]
    assert 'model="gt-6-as-a"' in configs
    assert 'model_reasoning_effort="xhigh"' in configs
    assert not any("web_search" in arg for arg in options), "Unexpected prior search binding"
    return command[:-1] + ["-c", SETTING, command[-1]]
