#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "GemmUKernel.h"
#include "StorageType.h"


namespace GEMMBench
{
    
struct BenchConfig
{
    size_t M = 32;
    size_t N = 32;
    size_t K = 32;

    size_t iterations = 100;
    size_t warmup = 10;

    size_t packet_size = 4;
    size_t lmul = 1;
    size_t alignment = 32;

    uint32_t seed = 12;

    double alpha = 1.0;
    double beta = 0.0;

    UKernelType kernel = UKernelType::IJK;
    PLayout storage_order = PLayout::Row;
};

inline BenchConfig& config()
{
    static BenchConfig cfg;
    return cfg;
}


inline void parseArguments(int argc,
                           char** argv)
{
    auto& cfg = config();

    for(int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];

        if(arg == "--packet-size" && i + 1 < argc)
        {
            cfg.packet_size = std::stoul(argv[++i]);
        }
        else if(arg == "--lmul" && i + 1 < argc)
        {
            cfg.lmul = std::stoul(argv[++i]);
        }
        else if(arg == "--alignment" && i + 1 < argc)
        {
            cfg.alignment = std::stoul(argv[++i]);
        }
        else if(arg == "--seed" && i + 1 < argc)
        {
            cfg.seed = std::stoul(argv[++i]);
        }
    }
}

}