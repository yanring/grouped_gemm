import os
from pathlib import Path
from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


# if not torch.cuda.is_available():
#     if os.environ.get("TORCH_CUDA_ARCH_LIST", None) is None:
#         os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"


cwd = Path(os.path.dirname(os.path.abspath(__file__)))
# _dc = torch.cuda.get_device_capability()
# _dc = f"{_dc[0]}{_dc[1]}"

# DEBUG: Currently we only test perf on A100
# _dc = 80

ext_modules = [
    CUDAExtension(
        "grouped_gemm_backend",
        ["csrc/ops.cu", "csrc/grouped_gemm.cu", "csrc/sinkhorn.cu", "csrc/permute.cu"],
        include_dirs = [
            f"{cwd}/third_party/cutlass/include/"
        ],
        extra_compile_args={
            "cxx": [
                "-fopenmp", "-fPIC", "-Wno-strict-aliasing"
            ],
            "nvcc": [
                f"--generate-code=arch=compute_80,code=sm_80",
                f"--generate-code=arch=compute_90,code=sm_90",
                f"-DGROUPED_GEMM_DEVICE_CAPABILITY=80",
                # NOTE: CUTLASS requires c++17.
                "-std=c++17",
            ],
        }
    )
]

setup(
    name="grouped_gemm",
    version="0.0.1",
    author="Trevor Gale, Shiqing Fan",
    author_email="tgale@stanford.edu, shiqingf@nvidia.com",
    description="GEMM Grouped",
    url="https://github.com/fanshiqing/grouped_gemm",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Unix",
    ],
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    install_requires=["absl-py", "numpy", "torch"],
)
