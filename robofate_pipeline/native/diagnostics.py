"""Original primary-exception diagnostics; no retry or qualification."""
import ast
import json
import os
from pathlib import Path
import sys
import time
import traceback

def reset_preserving_primary(client, scope, run):
    primary = sys.exc_info()[1]
    try:
        return client.call(func_name="reset_evaluation", obs=scope)
    except BaseException as cleanup:
        if primary is None:
            raise
        value = {
            "primary_type": type(primary).__name__, "primary_error": str(primary),
            "cleanup_type": type(cleanup).__name__, "cleanup_error": str(cleanup),
            "cleanup_traceback": traceback.format_exc(),
            "normal_shutdown_claimed": False, "primary_rethrown": True,
        }
        try:
            with (run / "final_reset_secondary_error.json").open("x") as stream:
                json.dump(value, stream, indent=2)
                stream.write("\n")
        except BaseException as diagnostic_error:
            print("Secondary-error diagnostic write failed: " + repr(diagnostic_error), file=sys.stderr)
            traceback.print_exception(type(cleanup), cleanup, cleanup.__traceback__, file=sys.stderr)


def bind_loop(original, raw, filename, run):
    tree = ast.parse(raw)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "eval_one_episode")
    text = ast.get_source_segment(raw.decode(), node)
    old = '    finally:\n        model_client.call(func_name="reset_evaluation", obs=scope)'
    assert text.count(old) == 1
    text = text.replace(old, "    finally:\n        _refate_final_reset(model_client, scope)", 1)
    namespace = dict(original.__globals__)
    namespace["_refate_final_reset"] = lambda client, scope: reset_preserving_primary(client, scope, run)
    exec(compile(text, str(filename) + "::preserve_primary_error", "exec"), namespace)
    return namespace["eval_one_episode"]
