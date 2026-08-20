# Copyright 2022 The IDEA Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------------------------
# Modified from
# https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/setup.py
# https://github.com/facebookresearch/detectron2/blob/main/setup.py
# https://github.com/open-mmlab/mmdetection/blob/master/setup.py
# https://github.com/Oneflow-Inc/libai/blob/main/setup.py
# ------------------------------------------------------------------------------------------------

import glob
import os
import subprocess

cwd = os.path.dirname(os.path.abspath(__file__))


def _pip_cuda_home() -> str:
    """CUDA toolkit root shipped by `cuda-toolkit[nvcc]` (nvidia/cuXX/{bin,include,lib})."""
    import nvidia

    homes = []
    for root in nvidia.__path__:
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            candidate = os.path.join(root, name)
            if os.path.isfile(os.path.join(candidate, "bin", "nvcc")):
                homes.append(candidate)
    assert homes, (
        "nvcc not found under the nvidia pip packages. "
        "Run `uv sync` (this project depends on cuda-toolkit[nvcc]) or set CUDA_HOME."
    )
    return homes[-1]


def _ensure_cuda_home() -> str:
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home is None:
        cuda_home = _pip_cuda_home()
    nvcc = os.path.join(cuda_home, "bin", "nvcc")
    assert os.path.isfile(nvcc), (
        f"nvcc not found at {nvcc}. Install cuda-toolkit[nvcc] (`uv sync`) or set CUDA_HOME "
        "to a CUDA toolkit that contains bin/nvcc."
    )
    os.environ["CUDA_HOME"] = cuda_home
    os.environ["PATH"] = os.path.join(cuda_home, "bin") + os.pathsep + os.environ.get("PATH", "")
    return cuda_home


_ensure_cuda_home()

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDA_HOME, CUDAExtension

# groundingdino version info
version = "0.1.0"
package_name = "groundingdino"

sha = "Unknown"
try:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd).decode("ascii").strip()
except Exception:
    pass


def write_version_file():
    version_path = os.path.join(cwd, "groundingdino", "version.py")
    with open(version_path, "w") as f:
        f.write(f"__version__ = '{version}'\n")


def get_extensions():
    assert CUDA_HOME is not None, "CUDA_HOME is None after _ensure_cuda_home()"
    extensions_dir = os.path.join(cwd, "groundingdino", "models", "GroundingDINO", "csrc")
    assert os.path.isdir(extensions_dir), f"GroundingDINO csrc not found: {extensions_dir}"

    sources = sorted(set(glob.glob(os.path.join(extensions_dir, "**", "*.cpp"), recursive=True)))
    source_cuda = sorted(set(glob.glob(os.path.join(extensions_dir, "**", "*.cu"), recursive=True)))
    assert sources, f"No C++ sources under {extensions_dir}"
    assert source_cuda, f"No CUDA sources under {extensions_dir}"

    cuda_lib = os.path.join(CUDA_HOME, "lib64")
    if not os.path.isdir(cuda_lib):
        cuda_lib = os.path.join(CUDA_HOME, "lib")
    assert os.path.isdir(cuda_lib), f"CUDA lib directory not found under {CUDA_HOME}"

    # pip cuda-toolkit ships libcudart.so.13 with no unversioned libcudart.so; the linker still
    # looks for -lcudart. Point at a local stub dir with those names.
    stub_dir = os.path.join(cwd, "build", "cuda_lib")
    os.makedirs(stub_dir, exist_ok=True)
    for name in os.listdir(cuda_lib):
        if ".so." not in name:
            continue
        unversioned = name.split(".so.")[0] + ".so"
        dest = os.path.join(stub_dir, unversioned)
        src = os.path.join(cuda_lib, name)
        if os.path.islink(dest) or os.path.isfile(dest):
            continue
        os.symlink(src, dest)

    print(f"Compiling GroundingDINO CUDA ops with CUDA_HOME={CUDA_HOME}")
    extra_compile_args = {
        "cxx": [],
        "nvcc": [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ],
    }
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    assert os.path.isdir(torch_lib), f"torch lib directory not found: {torch_lib}"
    rpath_args = [f"-Wl,-rpath,{torch_lib}", f"-Wl,-rpath,{cuda_lib}"]
    return [
        CUDAExtension(
            "groundingdino._C",
            sources + source_cuda,
            include_dirs=[extensions_dir],
            define_macros=[("WITH_CUDA", None)],
            extra_compile_args=extra_compile_args,
            extra_link_args=rpath_args,
            library_dirs=[stub_dir, cuda_lib, torch_lib],
        )
    ]


def parse_requirements(fname="requirements.txt", with_version=True):
    """Parse the package dependencies listed in a requirements file but strips
    specific versioning information.

    Args:
        fname (str): path to requirements file
        with_version (bool, default=False): if True include version specs

    Returns:
        List[str]: list of requirements items

    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
    """
    import re
    import sys
    from os.path import exists

    require_fpath = fname

    def parse_line(line):
        """Parse information from a line in a requirements text file."""
        if line.startswith("-r "):
            # Allow specifying requirements in other files
            target = line.split(" ")[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {"line": line}
            if line.startswith("-e "):
                info["package"] = line.split("#egg=")[1]
            elif "@git+" in line:
                info["package"] = line
            else:
                # Remove versioning from the package
                pat = "(" + "|".join([">=", "==", ">"]) + ")"
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info["package"] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ";" in rest:
                        # Handle platform specific dependencies
                        # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
                        version, platform_deps = map(str.strip, rest.split(";"))
                        info["platform_deps"] = platform_deps
                    else:
                        version = rest
                    info["version"] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    for info in parse_line(line):
                        yield info

    def gen_packages_items():
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info["package"]]
                if with_version and "version" in info:
                    parts.extend(info["version"])
                if not sys.version.startswith("3.4"):
                    # apparently package_deps are broken in 3.4
                    platform_deps = info.get("platform_deps")
                    if platform_deps is not None:
                        parts.append(";" + platform_deps)
                item = "".join(parts)
                yield item

    packages = list(gen_packages_items())
    return packages


if __name__ == "__main__":
    print(f"Building wheel {package_name}-{version}")
    os.chdir(cwd)

    license_path = os.path.join(cwd, "LICENSE")
    assert os.path.isfile(license_path), f"LICENSE not found: {license_path}"
    with open(license_path, "r", encoding="utf-8") as f:
        license_text = f.read()

    write_version_file()

    setup(
        name="groundingdino",
        version="0.1.0",
        author="International Digital Economy Academy, Shilong Liu",
        url="https://github.com/IDEA-Research/GroundingDINO",
        description="open-set object detector",
        license=license_text,
        install_requires=parse_requirements(os.path.join(cwd, "requirements.txt")),
        packages=find_packages(
            exclude=(
                "configs",
                "tests",
            )
        ),
        ext_modules=get_extensions(),
        cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
    )
