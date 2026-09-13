#pragma once 

namespace GEMMBench
{
    enum class PLayout
    {
        Row,
        Col
    };
    
    struct KernelPacking
    {
        PLayout A;
        PLayout B;
        PLayout C;
    };

}