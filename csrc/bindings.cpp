#include <torch/extension.h>

#include "cpu_fp32.h"

PYBIND11_MODULE(fmha, m)
{
    m.def("cpu_fmha_fp32", &cpu_fmha_fp32, "CPU FMHA F32");
}
