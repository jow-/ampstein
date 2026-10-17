#include "exl3_moe_coop_instances.cuh"
#include "../exl3_moe_coop_kernel.cuh"

// Mixed-width instances (ampstein): bits == 0 decodes every matrix at its own width, mul1 codebook only
MoeCoopKernel exl3_moe_coop_kernel_a_dyn(int Hi, bool wide)
{
    using namespace exl3_moe_coop_ns;
    const int smem = smem_a_bytes<0>(Hi);
    if (wide) return { (void*) exl3_moe_coop_a_kernel<0, 2, true>, smem };
    return { (void*) exl3_moe_coop_a_kernel<0, 2, false>, smem };
}

MoeCoopKernel exl3_moe_coop_kernel_b_dyn(bool wide)
{
    using namespace exl3_moe_coop_ns;
    const int smem = smem_b_bytes<0>();
    if (wide) return { (void*) exl3_moe_coop_b_kernel<0, 2, true>, smem };
    return { (void*) exl3_moe_coop_b_kernel<0, 2, false>, smem };
}
