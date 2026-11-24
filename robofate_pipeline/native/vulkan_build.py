"""Build only the owned compatibility library; never load Vulkan or a simulator."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess

from .. import evidence
from ..evidence import need
from ..runtime_policy import PACKAGE


def build(output, compiler, binding_ref):
    binding = evidence.decode(evidence.read_pin(binding_ref))
    need(binding.get("schema") == "robofate.vulkan-target.v1", "Explicit external build target required")
    for key in ("expected_hostname", "ssh_alias", "execution_owner"):
        need(type(binding[key]) is str and re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", binding[key]),
             "Bounded compile-time binding required")
    output, compiler = Path(output), Path(compiler)
    need(output.is_absolute() and output.parent.resolve(strict=True) == output.parent
         and not output.is_relative_to(PACKAGE.parent)
         and not PACKAGE.parent.is_relative_to(output), "Build output must be outside source")
    need(compiler.is_absolute() and compiler.resolve(strict=True) == compiler
         and os.access(compiler, os.X_OK), "Explicit canonical compiler required")
    output.mkdir(mode=0o700)
    (output / "build-evidence").mkdir(mode=0o700)
    source = PACKAGE / "native/vulkan"
    sources = {p.name: evidence.file_ref(p) for p in source.iterdir() if p.is_file()}
    for name in sources:
        with (output / name).open("xb") as stream:
            stream.write(evidence.read_pin(sources[name]))
    text = "#pragma once\n" + "".join(
        "#define " + macro + " " + json.dumps(binding[key]) + "\n"
        for macro, key in (("ROBOFATE_EXPECTED_HOST", "expected_hostname"),
                           ("ROBOFATE_EXPECTED_ALIAS", "ssh_alias"),
                           ("ROBOFATE_EXPECTED_OWNER", "execution_owner")))
    (output / "deployment_binding.hpp").write_text(text)
    library = output / "libVkLayer_REFATE_image_copy_staging.so"
    argv = [str(compiler), "-std=c++17", "-O2", "-shared", "-fPIC", "-pthread",
            "-Wl,-Bsymbolic", str(output / "copy_layer.cpp"), "-o", str(library)]
    evidence.exclusive(output / "BUILD_ARMED", dict(binding_ref=binding_ref, sources=sources,
        compiler_ref=evidence.file_ref(compiler), argv=argv, library_loaded=False))
    env = {k: v for k, v in os.environ.items() if k in ("PATH", "LANG", "LC_ALL", "TMPDIR")}
    with (output / "compiler.txt").open("x") as log:
        subprocess.run(argv, cwd=output, env=env, stdin=subprocess.DEVNULL, stdout=log,
                       stderr=subprocess.STDOUT, check=True, timeout=120)
    need(evidence.read_bytes(library)[:4] == b"\x7fELF", "Expected a Linux shared library")
    return evidence.exclusive(output / "build-evidence/BUILD.json", dict(
        schema="robofate.vulkan-build.v1", sources=sources, library_ref=evidence.file_ref(library),
        expected_hostname=binding["expected_hostname"], ssh_alias=binding["ssh_alias"],
        execution_owner=binding["execution_owner"], library_loaded=False, gpu_calls=0))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(build(args.output, args.compiler, {"path": str(args.binding), "sha256": args.sha256})))


if __name__ == "__main__":
    main()
