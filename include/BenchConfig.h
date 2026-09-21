#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
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

    bool csv_mode = false;
    std::string kernel_name = "ijk";
};

inline BenchConfig& config()
{
    static BenchConfig cfg;
    return cfg;
}


inline void parseArgs(int argc,
                      char** argv,
                      BenchConfig& cfg,
                      bool ignore_unknown = false)
{
    for(int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];

        auto next = [&]() {
            if (i + 1 >= argc)
            {
                std::cerr << "Missing value for argument: " << arg << '\n';
                std::exit(EXIT_FAILURE);
            }
            return argv[++i];
        };

        auto next_ulong = [&]() {
            return std::stoul(next());
        };

        auto next_double = [&]() {
            return std::stod(next());
        };

        if(arg == "--help" || arg == "-h")
        {
            std::cout << "Usage: GemmBench [options]\n"
                      << "Options:\n"
                      << "  --m <N>             M dimension (default: 32)\n"
                      << "  --n <N>             N dimension (default: 32)\n"
                      << "  --k <N>             K dimension (default: 32)\n"
                      << "  --alpha, -a <val>   Alpha parameter (default: 1.0)\n"
                      << "  --beta, -b <val>    Beta parameter (default: 0.0)\n"
                      << "  --kernel <name>     Kernel name (default: ijk)\n"
                      << "  --iterations <N>    Benchmark iterations (default: 100)\n"
                      << "  --warmup <N>        Warmup iterations (default: 10)\n"
                      << "  --packet-size <N>   SIMD packet size (default: 4)\n"
                      << "  --lmul <N>          RVV/MIPP LMUL factor (default: 1)\n"
                      << "  --alignment <N>     Memory alignment in bytes (default: 32)\n"
                      << "  --seed <N>          Random seed (default: 12)\n"
                      << "  --csv               Output in CSV format\n"
                      << "  --help, -h          Show this help message\n";
            std::exit(EXIT_SUCCESS);
        }
        else if(arg == "--m")
            cfg.M = next_ulong();
        else if(arg == "--n")
            cfg.N = next_ulong();
        else if(arg == "--k")
            cfg.K = next_ulong();
        else if(arg == "--alpha" || arg == "-a")
            cfg.alpha = next_double();
        else if(arg == "--beta" || arg == "-b")
            cfg.beta = next_double();
        else if(arg == "--iterations")
            cfg.iterations = next_ulong();
        else if(arg == "--warmup")
            cfg.warmup = next_ulong();
        else if(arg == "--packet-size")
            cfg.packet_size = next_ulong();
        else if(arg == "--lmul")
            cfg.lmul = next_ulong();
        else if(arg == "--alignment")
            cfg.alignment = next_ulong();
        else if(arg == "--seed")
            cfg.seed = static_cast<uint32_t>(next_ulong());
        else if(arg == "--kernel")
        {
            cfg.kernel_name = next();
            cfg.kernel = getKernelDescriptor(cfg.kernel_name).type;
        }
        else if(arg == "--csv")
            cfg.csv_mode = true;
        else if(!ignore_unknown)
        {
            std::cerr << "Unknown argument: " << arg << '\n';
            std::exit(EXIT_FAILURE);
        }
    }
}

inline BenchConfig parseArgs(int argc,
                             char** argv,
                             bool ignore_unknown = false)
{
    BenchConfig cfg;
    parseArgs(argc, argv, cfg, ignore_unknown);
    return cfg;
}

inline void parseArguments(int argc,
                           char** argv)
{
    parseArgs(argc, argv, config(), /*ignore_unknown=*/true);
}

} // namespace GEMMBench