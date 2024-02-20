#include "grouped_gemm.h"
#include "permute.h"
#include "sinkhorn.h"

#include <torch/extension.h>

namespace grouped_gemm {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gmm", &GroupedGemm, "Grouped GEMM.");
  m.def("sinkhorn", &sinkhorn, "Sinkhorn kernel");
  m.def("permute", &moe_permute_op, "Token permutation kernel");
  m.def("unpermute", &moe_recover_op, "Token un-permutation kernel");
}

}  // namespace grouped_gemm
