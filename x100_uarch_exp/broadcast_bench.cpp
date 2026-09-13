#include <chrono>
#include <cstdio>
#include <mipp.hpp>


using T = double;


static inline double run_pure_fma(size_t iterations)
{
    using namespace mipp;

    constexpr int lmul = 1;

    auto a = set1<T,lmul>(1.001);
    auto b = set1<T,lmul>(1.002);


    auto c0  = set0<T,lmul>();
    auto c1  = set0<T,lmul>();
    auto c2  = set0<T,lmul>();
    auto c3  = set0<T,lmul>();

    auto c4  = set0<T,lmul>();
    auto c5  = set0<T,lmul>();
    auto c6  = set0<T,lmul>();
    auto c7  = set0<T,lmul>();

    auto c8  = set0<T,lmul>();
    auto c9  = set0<T,lmul>();
    auto c10 = set0<T,lmul>();
    auto c11 = set0<T,lmul>();

    auto c12 = set0<T,lmul>();
    auto c13 = set0<T,lmul>();
    auto c14 = set0<T,lmul>();
    auto c15 = set0<T,lmul>();


    auto start = std::chrono::high_resolution_clock::now();


    for(size_t i = 0; i < iterations; ++i)
    {
        c0  = fmadd(a,b,c0);
        c1  = fmadd(a,b,c1);
        c2  = fmadd(a,b,c2);
        c3  = fmadd(a,b,c3);

        c4  = fmadd(a,b,c4);
        c5  = fmadd(a,b,c5);
        c6  = fmadd(a,b,c6);
        c7  = fmadd(a,b,c7);

        c8  = fmadd(a,b,c8);
        c9  = fmadd(a,b,c9);
        c10 = fmadd(a,b,c10);
        c11 = fmadd(a,b,c11);

        c12 = fmadd(a,b,c12);
        c13 = fmadd(a,b,c13);
        c14 = fmadd(a,b,c14);
        c15 = fmadd(a,b,c15);
    }


    auto end = std::chrono::high_resolution_clock::now();


    double checksum = 0;

    alignas(64) T tmp[N<T,lmul>()];

    store(tmp, c0);
    checksum += tmp[0];

    store(tmp, c1);
    checksum += tmp[0];

    store(tmp, c2);
    checksum += tmp[0];

    store(tmp, c3);
    checksum += tmp[0];

    store(tmp, c4);
    checksum += tmp[0];

    store(tmp, c5);
    checksum += tmp[0];

    store(tmp, c6);
    checksum += tmp[0];

    store(tmp, c7);
    checksum += tmp[0];

    store(tmp, c8);
    checksum += tmp[0];

    store(tmp, c9);
    checksum += tmp[0];

    store(tmp, c10);
    checksum += tmp[0];

    store(tmp, c11);
    checksum += tmp[0];

    store(tmp, c12);
    checksum += tmp[0];

    store(tmp, c13);
    checksum += tmp[0];

    store(tmp, c14);
    checksum += tmp[0];

    store(tmp, c15);
    checksum += tmp[0];


    double time =
        std::chrono::duration<double>(end-start).count();


    constexpr double flops_per_iter = 16 * N<T,lmul>() * 2;

    printf("checksum=%lf\n", checksum);
    printf("duration=%lf\n", time);
    return (iterations * flops_per_iter) / time / 1e9;
}



static inline double run_broadcast_fma(size_t iterations)
{
    using namespace mipp;

    constexpr int lmul = 1;


    auto b = set1<T,lmul>(1.002);


    auto c0  = set0<T,lmul>();
    auto c1  = set0<T,lmul>();
    auto c2  = set0<T,lmul>();
    auto c3  = set0<T,lmul>();

    auto c4  = set0<T,lmul>();
    auto c5  = set0<T,lmul>();
    auto c6  = set0<T,lmul>();
    auto c7  = set0<T,lmul>();

    auto c8  = set0<T,lmul>();
    auto c9  = set0<T,lmul>();
    auto c10 = set0<T,lmul>();
    auto c11 = set0<T,lmul>();

    auto c12 = set0<T,lmul>();
    auto c13 = set0<T,lmul>();
    auto c14 = set0<T,lmul>();
    auto c15 = set0<T,lmul>();

    auto c16 = set0<T>();
    auto c17 = set0<T>();
    auto c18 = set0<T>();
    auto c19 = set0<T>();
    auto c20 = set0<T>();
    auto c21 = set0<T>();
    auto c22 = set0<T>();
    auto c23 = set0<T>();



    double scalar = 1.001;


    auto start = std::chrono::high_resolution_clock::now();


    for(size_t i = 0; i < iterations; ++i)
    {
        auto a = set1<T,lmul>(scalar);
        scalar += 0.000001;


        c0  = fmadd(a,b,c0);
        c1  = fmadd(a,b,c1);
        c2  = fmadd(a,b,c2);
        c3  = fmadd(a,b,c3);

        c4  = fmadd(a,b,c4);
        c5  = fmadd(a,b,c5);
        c6  = fmadd(a,b,c6);
        c7  = fmadd(a,b,c7);

        c8  = fmadd(a,b,c8);
        c9  = fmadd(a,b,c9);
        c10 = fmadd(a,b,c10);
        c11 = fmadd(a,b,c11);

        c12 = fmadd(a,b,c12);
        c13 = fmadd(a,b,c13);
        c14 = fmadd(a,b,c14);
        c15 = fmadd(a,b,c15);

        c16 = fmadd(a,b,c16);
        c17 = fmadd(a,b,c17);
        c18 = fmadd(a,b,c18);
        c19 = fmadd(a,b,c19);
        c20 = fmadd(a,b,c20);
        c21 = fmadd(a,b,c21);
        c22 = fmadd(a,b,c22);
        c23 = fmadd(a,b,c23);
    }


    auto end = std::chrono::high_resolution_clock::now();


    double checksum = 0;

    alignas(64) T tmp[N<T,lmul>()];

    store(tmp, c0);
    checksum += tmp[0];

    store(tmp, c1);
    checksum += tmp[0];

    store(tmp, c2);
    checksum += tmp[0];

    store(tmp, c3);
    checksum += tmp[0];

    store(tmp, c4);
    checksum += tmp[0];

    store(tmp, c5);
    checksum += tmp[0];

    store(tmp, c6);
    checksum += tmp[0];

    store(tmp, c7);
    checksum += tmp[0];

    store(tmp, c8);
    checksum += tmp[0];

    store(tmp, c9);
    checksum += tmp[0];

    store(tmp, c10);
    checksum += tmp[0];

    store(tmp, c11);
    checksum += tmp[0];

    store(tmp, c12);
    checksum += tmp[0];

    store(tmp, c13);
    checksum += tmp[0];

    store(tmp, c14);
    checksum += tmp[0];

    store(tmp, c15);
    checksum += tmp[0];

    store(tmp, c16);
    checksum += tmp[0];
    store(tmp, c17);
    checksum += tmp[0];
    store(tmp, c18);
    checksum += tmp[0];
    store(tmp, c19);
    checksum += tmp[0];
    store(tmp, c20);
    checksum += tmp[0];
    store(tmp, c21);
    checksum += tmp[0];
    store(tmp, c22);
    checksum += tmp[0];
    store(tmp, c23);
    checksum += tmp[0];


    double time =
        std::chrono::duration<double>(end-start).count();


    constexpr double flops_per_iter = 24 * N<T,lmul>() * 2;

    printf("checksum=%lf\n", checksum);
    printf("duration=%lf\n", time);

    return (iterations * flops_per_iter) / time / 1e9;
}


template<typename T>
static inline
double benchmark_load_fma(size_t iterations)
{
    using namespace mipp;

    constexpr size_t STRIDE = 64;

    alignas(64) T dummy_a[STRIDE * N<T>()];
    alignas(64) T dummy_b[N<T>()];


    //
    // Initialize dummy data
    //
    for(size_t i = 0; i < STRIDE * N<T>(); ++i)
        dummy_a[i] = static_cast<T>(i + 1);

    for(size_t i = 0; i < N<T>(); ++i)
        dummy_b[i] = static_cast<T>(1.0 + i);


    auto b = load<T>(dummy_b);


    auto c0  = set0<T>();
    auto c1  = set0<T>();
    auto c2  = set0<T>();
    auto c3  = set0<T>();

    auto c4  = set0<T>();
    auto c5  = set0<T>();
    auto c6  = set0<T>();
    auto c7  = set0<T>();

    auto c8  = set0<T>();
    auto c9  = set0<T>();
    auto c10 = set0<T>();
    auto c11 = set0<T>();

    auto c12 = set0<T>();
    auto c13 = set0<T>();
    auto c14 = set0<T>();
    auto c15 = set0<T>();

    // up to 24 fma

    auto c16 = set0<T>();
    auto c17 = set0<T>();
    auto c18 = set0<T>();
    auto c19 = set0<T>();
    auto c20 = set0<T>();
    auto c21 = set0<T>();
    auto c22 = set0<T>();
    auto c23 = set0<T>();


    size_t offset = 0;

    auto start = std::chrono::high_resolution_clock::now();

    for(size_t i = 0; i < iterations; ++i)
    {
        auto a = load<T>(dummy_a + offset);

        offset += N<T>();

        if(offset >= STRIDE * N<T>())
            offset = 0;


        c0  = fmadd(a,b,c0);
        c1  = fmadd(a,b,c1);
        c2  = fmadd(a,b,c2);
        c3  = fmadd(a,b,c3);

        c4  = fmadd(a,b,c4);
        c5  = fmadd(a,b,c5);
        c6  = fmadd(a,b,c6);
        c7  = fmadd(a,b,c7);

        c8  = fmadd(a,b,c8);
        c9  = fmadd(a,b,c9);
        c10 = fmadd(a,b,c10);
        c11 = fmadd(a,b,c11);

        c12 = fmadd(a,b,c12);
        c13 = fmadd(a,b,c13);
        c14 = fmadd(a,b,c14);
        c15 = fmadd(a,b,c15);

        c16 = fmadd(a,b,c16);
        c17 = fmadd(a,b,c17);
        c18 = fmadd(a,b,c18);
        c19 = fmadd(a,b,c19);

        c20 = fmadd(a,b,c20);
        c21 = fmadd(a,b,c21);
        c22 = fmadd(a,b,c22);
        c23 = fmadd(a,b,c23);
    }

    auto end = std::chrono::high_resolution_clock::now();

    //
    // Checksum
        //
    // Checksum
    //
    auto sum = c0;

    sum = add(sum, c1);
    sum = add(sum, c2);
    sum = add(sum, c3);

    sum = add(sum, c4);
    sum = add(sum, c5);
    sum = add(sum, c6);
    sum = add(sum, c7);

    sum = add(sum, c8);
    sum = add(sum, c9);
    sum = add(sum, c10);
    sum = add(sum, c11);

    sum = add(sum, c12);
    sum = add(sum, c13);
    sum = add(sum, c14);
    sum = add(sum, c15);

    sum = add(sum, c16);
    sum = add(sum, c17);
    sum = add(sum, c18);
    sum = add(sum, c19);
    sum = add(sum, c20);
    sum = add(sum, c21);
    sum = add(sum, c22);
    sum = add(sum, c23);


    alignas(64) T result[N<T>()];

    store(result, sum);


    double checksum = 0.0;

    for(size_t i = 0; i < N<T>(); ++i)
        checksum += result[i];


    printf("checksum=%lf\n", checksum);

    auto time =
        std::chrono::duration<double>(end-start).count();
    double flops_per_iter = 24 * N<T>() * 2;

    printf("duration=%lf\n", time);

    return (iterations * flops_per_iter) / time / 1e9;
}

template<typename T>
static inline
double benchmark_load_fma_lmul4(size_t iterations)
{
    using namespace mipp;

    constexpr size_t lmul = 4;


    alignas(64) volatile T dummy_a[N<T,lmul>()];
    alignas(64) T dummy_b[N<T,lmul>()];

    //
    // Initialize dummy data
    //
    for(size_t i = 0; i < N<T,lmul>(); ++i)
    {
        const_cast<T*>(dummy_a)[i] = static_cast<T>(i + 1);
        dummy_b[i] = static_cast<T>(1.0 + i);
    }

    auto b = load<T,lmul>(dummy_b);

    //
    // LMUL=4:
    // 1 vector = 4 architectural RVV registers
    //
    // b      : 4 registers
    // c0-c5  : 24 registers
    //
    // Leaves room for temporaries
    //
    auto c0 = set0<T,lmul>();
    auto c1 = set0<T,lmul>();
    auto c2 = set0<T,lmul>();

    auto c3 = set0<T,lmul>();
    auto c4 = set0<T,lmul>();
    auto c5 = set0<T,lmul>();


    auto start = std::chrono::high_resolution_clock::now();


    for(size_t i = 0; i < iterations; ++i)
    {

        asm volatile("" ::: "memory");

        auto a = load<T,lmul>(
            const_cast<T*>(dummy_a)
        );

        asm volatile("" ::: "memory");



        c0 = fmadd(a,b,c0);
        c1 = fmadd(a,b,c1);
        c2 = fmadd(a,b,c2);

        c3 = fmadd(a,b,c3);
        c4 = fmadd(a,b,c4);
        c5 = fmadd(a,b,c5);
    }


    auto end = std::chrono::high_resolution_clock::now();


    //
    // Checksum
    //
    auto sum = c0;

    sum = add(sum,c1);
    sum = add(sum,c2);
    sum = add(sum,c3);
    sum = add(sum,c4);
    sum = add(sum,c5);


    alignas(64) T result[N<T,lmul>()];

    store(result,sum);


    double checksum = 0.0;

    for(size_t i = 0; i < N<T,lmul>(); ++i)
        checksum += result[i];


    printf("checksum=%lf\n", checksum);


    double time =
        std::chrono::duration<double>(end-start).count();
    printf("duration=%lf\n", time);


    constexpr double flops_per_iter =
        6 * N<T,lmul>() * 2;


    return (iterations * flops_per_iter)
            / time
            / 1e9;
}

template<typename T>
static inline
double benchmark_load_fma_lmul4_2loads(size_t iterations)
{
    using namespace mipp;

    constexpr int lmul = 4;

    alignas(64) T dummy_a0[N<T,lmul>()];
    alignas(64) T dummy_a1[N<T,lmul>()];
    alignas(64) T dummy_b[N<T,lmul>()];


    for(size_t i = 0; i < N<T,lmul>(); ++i)
    {
        dummy_a0[i] = static_cast<T>(i + 1);
        dummy_a1[i] = static_cast<T>(i + 2);
        dummy_b[i]  = static_cast<T>(1.0 + i);
    }


    auto b = load<T,lmul>(dummy_b);


    //
    // 5 LMUL=4 accumulators
    //
    auto c0 = set0<T,lmul>();
    auto c1 = set0<T,lmul>();
    auto c2 = set0<T,lmul>();
    auto c3 = set0<T,lmul>();
    auto c4 = set0<T,lmul>();


    auto start = std::chrono::high_resolution_clock::now();


    for(size_t i = 0; i < iterations; ++i)
    {

        asm volatile("" ::: "memory");
        auto a0 = load<T,lmul>(dummy_a0);
        asm volatile("" ::: "memory");

        asm volatile("" ::: "memory");
        auto a1 = load<T,lmul>(dummy_a1);
        asm volatile("" ::: "memory");

        c0 = fmadd(a0,b,c0);
        c1 = fmadd(a0,b,c1);
        c2 = fmadd(a0,b,c2);


        c3 = fmadd(a1,b,c3);
        c4 = fmadd(a1,b,c4);
    }


    auto end = std::chrono::high_resolution_clock::now();


    //
    // Checksum
    //
    auto sum = c0;

    sum = add(sum,c1);
    sum = add(sum,c2);
    sum = add(sum,c3);
    sum = add(sum,c4);


    alignas(64) T result[N<T,lmul>()];

    store(result,sum);


    double checksum = 0.0;

    for(size_t i = 0; i < N<T,lmul>(); ++i)
        checksum += result[i];


    printf("checksum=%lf\n", checksum);


    double time =
        std::chrono::duration<double>(end-start).count();
    printf("duration=%lf\n", time);

    //
    // 5 FMAs per loaded A vector, 2 A vectors loaded
    //
    constexpr double flops_per_iter =
        5 * N<T,lmul>() * 2;


    return (iterations * flops_per_iter)
            / time
            / 1e9;
}

template<typename T>
static inline
double benchmark_load_fma_lmul4_3loads(size_t iterations)
{
    using namespace mipp;

    constexpr int lmul = 4;


    alignas(64) T dummy_a0[N<T,lmul>()];
    alignas(64) T dummy_a1[N<T,lmul>()];
    alignas(64) T dummy_a2[N<T,lmul>()];
    alignas(64) T dummy_b[N<T,lmul>()];


    for(size_t i = 0; i < N<T,lmul>(); ++i)
    {
        dummy_a0[i] = static_cast<T>(i + 1);
        dummy_a1[i] = static_cast<T>(i + 2);
        dummy_a2[i] = static_cast<T>(i + 3);
        dummy_b[i]  = static_cast<T>(1.0 + i);
    }


    auto b = load<T,lmul>(dummy_b);


    auto c0 = set0<T,lmul>();
    auto c1 = set0<T,lmul>();
    auto c2 = set0<T,lmul>();
    auto c3 = set0<T,lmul>();


    auto start = std::chrono::high_resolution_clock::now();


    for(size_t i = 0; i < iterations; ++i)
    {
        asm volatile("" ::: "memory");
        auto a0 = load<T,lmul>(dummy_a0);
        asm volatile("" ::: "memory");

        asm volatile("" ::: "memory");
        auto a1 = load<T,lmul>(dummy_a1);
        asm volatile("" ::: "memory");

        asm volatile("" ::: "memory");
        auto a2 = load<T,lmul>(dummy_a2);
        asm volatile("" ::: "memory");


        c0 = fmadd(a0,b,c0);
        c1 = fmadd(a0,b,c1);


        c2 = fmadd(a1,b,c2);


        c3 = fmadd(a2,b,c3);
    }


    auto end = std::chrono::high_resolution_clock::now();


    auto sum = c0;

    sum = add(sum,c1);
    sum = add(sum,c2);
    sum = add(sum,c3);


    alignas(64) T result[N<T,lmul>()];

    store(result,sum);


    double checksum = 0.0;

    for(size_t i = 0; i < N<T,lmul>(); ++i)
        checksum += result[i];


    printf("checksum=%lf\n", checksum);


    double time =
        std::chrono::duration<double>(end-start).count();

    printf("duration=%lf\n", time);

    //
    // 4 FMAs * VL
    //
    constexpr double flops_per_iter =
        4 * N<T,lmul>() * 2;


    return (iterations * flops_per_iter)
            / time
            / 1e9;
}

template<typename T>
static inline
double benchmark_fma_lmul4_2broadcasts(size_t iterations)
{
    using namespace mipp;

    constexpr int lmul = 4;


    alignas(64) T dummy_b[N<T,lmul>()];
    alignas(64) T dummy_c[6][N<T,lmul>()];


    for(size_t i = 0; i < N<T,lmul>(); ++i)
    {
        dummy_b[i] = static_cast<T>(1.0 + i);
    }


    for(size_t i = 0; i < 6; ++i)
    {
        for(size_t j = 0; j < N<T,lmul>(); ++j)
            dummy_c[i][j] = 0;
    }


    auto b = load<T,lmul>(dummy_b);


    auto c0 = set0<T,lmul>();
    auto c1 = set0<T,lmul>();
    auto c2 = set0<T,lmul>();
    auto c3 = set0<T,lmul>();

    auto c4 = set0<T,lmul>();
    auto c5 = set0<T,lmul>();



    double scalar0 = 1.001;
    double scalar1 = 2.001;


    auto start = std::chrono::high_resolution_clock::now();


    for(size_t i = 0; i < iterations; ++i)
    {
        auto a0 = set1<T,lmul>(scalar0);
        auto a1 = set1<T,lmul>(scalar1);

        scalar0 += 0.000001;
        scalar1 += 0.000002;


        c0 = fmadd(a0,b,c0);
        c1 = fmadd(a0,b,c1);
        c2 = fmadd(a0,b,c2);
        c3 = fmadd(a0,b,c3);

        c4 = fmadd(a1,b,c4);
        c5 = fmadd(a1,b,c5);

    }


    auto end = std::chrono::high_resolution_clock::now();


    //
    // Force stores
    //
    store(dummy_c[0], c0);
    store(dummy_c[1], c1);
    store(dummy_c[2], c2);
    store(dummy_c[3], c3);

    store(dummy_c[4], c4);
    store(dummy_c[5], c5);



    double checksum = 0;

    for(size_t i = 0; i < 6; ++i)
        checksum += dummy_c[i][0];


    printf("checksum=%lf\n", checksum);


    double time =
        std::chrono::duration<double>(end-start).count();

    printf("duration=%lf\n", time);

    //
    // 6 accumulators * VL * 2 FLOPs
    //
    constexpr double flops_per_iter =
        6 * N<T,lmul>() * 2;


    return (iterations * flops_per_iter)
            / time
            / 1e9;
}

template<typename T>
static inline
double benchmark_fma_lmul4_broadcasts(size_t iterations)
{
    using namespace mipp;

    constexpr int lmul = 4;

    alignas(64) T dummy_b[N<T,lmul>()];

    for(size_t i = 0; i < N<T,lmul>(); ++i)
        dummy_b[i] = static_cast<T>(1.0 + i);


    //
    // One B vector
    //
    auto b = load<T,lmul>(dummy_b);


    //
    // Six accumulators
    //
    auto c0 = set0<T,lmul>();
    auto c1 = set0<T,lmul>();
    auto c2 = set0<T,lmul>();
    auto c3 = set0<T,lmul>();
    auto c4 = set0<T,lmul>();
    auto c5 = set0<T,lmul>();


    double scalar = 1.001;


    auto start = std::chrono::high_resolution_clock::now();


    for(size_t i = 0; i < iterations; ++i)
    {
        //
        // One broadcast
        //
        auto a = set1<T,lmul>(scalar);

        scalar += 0.000001;


        //
        // Six independent accumulation chains
        //
        c0 = fmadd(a,b,c0);
        c1 = fmadd(a,b,c1);
        c2 = fmadd(a,b,c2);

        c3 = fmadd(a,b,c3);
        c4 = fmadd(a,b,c4);
        c5 = fmadd(a,b,c5);
    }


    auto end = std::chrono::high_resolution_clock::now();


    //
    // Checksum
    //
    auto sum = c0;

    sum = add(sum,c1);
    sum = add(sum,c2);
    sum = add(sum,c3);
    sum = add(sum,c4);
    sum = add(sum,c5);


    alignas(64) T result[N<T,lmul>()];

    store(result,sum);


    double checksum = 0.0;

    for(size_t i = 0; i < N<T,lmul>(); ++i)
        checksum += result[i];


    printf("checksum=%lf\n", checksum);


    double time =
        std::chrono::duration<double>(end-start).count();


    constexpr double flops_per_iter =
        6 * N<T,lmul>() * 2;


    return (iterations * flops_per_iter)
        / time
        / 1e9;
}

template<typename T>
static inline
double benchmark_gemm_pattern_lmul4(
    size_t iterations)
{
    using namespace mipp;

    constexpr int lmul = 4;
    constexpr size_t VL = N<T,lmul>();

    alignas(64) T dummy_a[2];
    alignas(64) T dummy_b[3*VL];

    dummy_a[0] = 1.001;
    dummy_a[1] = 2.001;

    for(size_t i = 0; i < 3*VL; ++i)
        dummy_b[i] = 1.0 + i;


    auto c00 = set0<T,lmul>();
    auto c01 = set0<T,lmul>();
    auto c02 = set0<T,lmul>();

    auto c10 = set0<T,lmul>();
    auto c11 = set0<T,lmul>();
    auto c12 = set0<T,lmul>();


    auto start =
        std::chrono::high_resolution_clock::now();


    for(size_t it = 0; it < iterations; ++it)
    {
        auto a0 = set1<T,lmul>(dummy_a[0]);
        auto a1 = set1<T,lmul>(dummy_a[1]);


        auto b0 = load<T,lmul>(dummy_b + 0*VL);

        c00 = fmadd(a0,b0,c00);
        c10 = fmadd(a1,b0,c10);


        auto b1 = load<T,lmul>(dummy_b + 1*VL);

        c01 = fmadd(a0,b1,c01);
        c11 = fmadd(a1,b1,c11);


        auto b2 = load<T,lmul>(dummy_b + 2*VL);

        c02 = fmadd(a0,b2,c02);
        c12 = fmadd(a1,b2,c12);
    }


    auto end =
        std::chrono::high_resolution_clock::now();


    //
    // checksum
    //
    auto sum = c00;

    sum = add(sum,c01);
    sum = add(sum,c02);

    sum = add(sum,c10);
    sum = add(sum,c11);
    sum = add(sum,c12);


    alignas(64) T result[VL];

    store(result,sum);

    double checksum = 0;

    for(size_t i = 0; i < VL; ++i)
        checksum += result[i];


    printf("checksum=%lf\n", checksum);


    double time =
        std::chrono::duration<double>(end-start).count();
    printf("duration=%lf\n", time);

    constexpr double flops_per_iter =
        2 * VL * 6;


    return iterations * flops_per_iter
        / time
        / 1e9;
}

template<typename T>
static inline
double benchmark_gemm_pattern_lmul4_mr1_nr6(size_t iterations)
{
    using namespace mipp;

    constexpr int lmul = 4;
    constexpr size_t VL = N<T,lmul>();
    constexpr size_t NR = 6 * VL;


    alignas(64) T dummy_a[1];
    alignas(64) T dummy_b[NR];


    dummy_a[0] = static_cast<T>(1.001);

    for(size_t i = 0; i < NR; ++i)
        dummy_b[i] = static_cast<T>(1.0 + i);


    auto c0 = set0<T,lmul>();
    auto c1 = set0<T,lmul>();
    auto c2 = set0<T,lmul>();

    auto c3 = set0<T,lmul>();
    auto c4 = set0<T,lmul>();
    auto c5 = set0<T,lmul>();


    auto start =
        std::chrono::high_resolution_clock::now();


    for(size_t it = 0; it < iterations; ++it)
    {
        //
        // One A broadcast
        //
        auto a = set1<T,lmul>(dummy_a[0]);


        //
        // Six B vectors
        //
        auto b0 = load<T,lmul>(dummy_b + 0*VL);
        c0 = fmadd(a,b0,c0);


        auto b1 = load<T,lmul>(dummy_b + 1*VL);
        c1 = fmadd(a,b1,c1);


        auto b2 = load<T,lmul>(dummy_b + 2*VL);
        c2 = fmadd(a,b2,c2);


        auto b3 = load<T,lmul>(dummy_b + 3*VL);
        c3 = fmadd(a,b3,c3);


        auto b4 = load<T,lmul>(dummy_b + 4*VL);
        c4 = fmadd(a,b4,c4);


        auto b5 = load<T,lmul>(dummy_b + 5*VL);
        c5 = fmadd(a,b5,c5);
    }


    auto end =
        std::chrono::high_resolution_clock::now();


    //
    // checksum
    //
    auto sum = c0;

    sum = add(sum,c1);
    sum = add(sum,c2);
    sum = add(sum,c3);
    sum = add(sum,c4);
    sum = add(sum,c5);


    alignas(64) T result[VL];

    store(result,sum);


    double checksum = 0.0;

    for(size_t i = 0; i < VL; ++i)
        checksum += result[i];


    printf("checksum=%lf\n", checksum);


    double time =
        std::chrono::duration<double>(end-start).count();

    printf("duration=%lf\n", time);
    constexpr double flops_per_iter =
        6 * VL * 2;


    return (iterations * flops_per_iter)
        / time
        / 1e9;
}

template<typename T>
static inline
double benchmark_gemm_pattern_lmul4_mr1_nr5(size_t iterations)
{
    using namespace mipp;

    constexpr int lmul = 4;
    constexpr size_t VL = N<T,lmul>();

    constexpr size_t NR = 5 * VL;


    alignas(64) T dummy_a[1];
    alignas(64) T dummy_b[NR];


    dummy_a[0] = static_cast<T>(1.001);

    for(size_t i = 0; i < NR; ++i)
        dummy_b[i] = static_cast<T>(1.0 + i);


    auto c0 = set0<T,lmul>();
    auto c1 = set0<T,lmul>();
    auto c2 = set0<T,lmul>();
    auto c3 = set0<T,lmul>();
    auto c4 = set0<T,lmul>();


    auto start =
        std::chrono::high_resolution_clock::now();


    for(size_t it = 0; it < iterations; ++it)
    {
        //
        // A row broadcast
        //
        auto a = set1<T,lmul>(dummy_a[0]);


        auto b0 = load<T,lmul>(dummy_b + 0*VL);
        c0 = fmadd(a,b0,c0);


        auto b1 = load<T,lmul>(dummy_b + 1*VL);
        c1 = fmadd(a,b1,c1);


        auto b2 = load<T,lmul>(dummy_b + 2*VL);
        c2 = fmadd(a,b2,c2);


        auto b3 = load<T,lmul>(dummy_b + 3*VL);
        c3 = fmadd(a,b3,c3);


        auto b4 = load<T,lmul>(dummy_b + 4*VL);
        c4 = fmadd(a,b4,c4);
    }


    auto end =
        std::chrono::high_resolution_clock::now();


    //
    // checksum
    //
    auto sum = c0;

    sum = add(sum,c1);
    sum = add(sum,c2);
    sum = add(sum,c3);
    sum = add(sum,c4);


    alignas(64) T result[VL];

    store(result,sum);


    double checksum = 0.0;

    for(size_t i = 0; i < VL; ++i)
        checksum += result[i];


    printf("checksum=%lf\n", checksum);


    double time =
        std::chrono::duration<double>(end-start).count();

    printf("duration=%lf\n", time);

    constexpr double flops_per_iter =
        5 * VL * 2;


    return (iterations * flops_per_iter)
        / time
        / 1e9;
}

template<typename T>
static inline
double benchmark_gemm_pattern_lmul8(size_t iterations)
{
    using namespace mipp;

    constexpr int lmul = 8;
    constexpr size_t VL = N<T,lmul>();
    constexpr size_t STRIDE = 8;


    alignas(64) T dummy_a[STRIDE];
    alignas(64) T dummy_b[STRIDE][VL];


    for(size_t i = 0; i < STRIDE; ++i)
    {
        dummy_a[i] = static_cast<T>(1.001 + i);

        for(size_t j = 0; j < VL; ++j)
            dummy_b[i][j] = static_cast<T>(1.0 + i + j);
    }


    auto c = set0<T,lmul>();


    size_t offset = 0;


    auto start =
        std::chrono::high_resolution_clock::now();


    for(size_t it = 0; it < iterations; ++it)
    {

        auto a = set1<T,lmul>(dummy_a[offset]); 
        auto b = load<T,lmul>(dummy_b[offset]);


        c = fmadd(a,b,c);


        offset++;

        if(offset == STRIDE)
            offset = 0;
    }


    auto end =
        std::chrono::high_resolution_clock::now();


    alignas(64) T result[VL];

    store(result,c);


    double checksum = 0.0;

    for(size_t i = 0; i < VL; ++i)
        checksum += result[i];


    printf("checksum=%lf\n", checksum);


    double time =
        std::chrono::duration<double>(end-start).count();


    printf("duration=%lf\n", time);

    constexpr double flops_per_iter =
        VL * 2;


    return (iterations * flops_per_iter)
        / time
        / 1e9;
}

template<typename T>
static inline
double benchmark_gemm_pattern_lmul8_nr2(size_t iterations)
{
    using namespace mipp;

    constexpr int lmul = 8;
    constexpr size_t VL = N<T,lmul>();


    alignas(64) T dummy_a[1];
    alignas(64) T dummy_b[2 * VL];


    dummy_a[0] = static_cast<T>(1.001);

    for(size_t i = 0; i < 2 * VL; ++i)
        dummy_b[i] = static_cast<T>(1.0 + i);


    auto c0 = set0<T,lmul>();
    auto c1 = set0<T,lmul>();


    auto start =
        std::chrono::high_resolution_clock::now();


    for(size_t it = 0; it < iterations; ++it)
    {
        auto a = set1<T,lmul>(dummy_a[0]);


        auto b0 = load<T,lmul>(dummy_b);
        auto b1 = load<T,lmul>(dummy_b + VL);


        c0 = fmadd(a,b0,c0);
        c1 = fmadd(a,b1,c1);
    }


    auto end =
        std::chrono::high_resolution_clock::now();


    //
    // checksum
    //
    auto sum = add(c0,c1);


    alignas(64) T result[VL];

    store(result,sum);


    double checksum = 0.0;

    for(size_t i = 0; i < VL; ++i)
        checksum += result[i];


    printf("checksum=%lf\n", checksum);


    double time =
        std::chrono::duration<double>(end-start).count();

    printf("duration=%lf\n", time);
    constexpr double flops_per_iter =
        2 * VL * 2;


    return (iterations * flops_per_iter)
        / time
        / 1e9;
}

template<typename T>
static inline
double benchmark_gemm_pattern_lmul4_nr3(size_t iterations)
{
    using namespace mipp;

    constexpr int lmul = 4;
    constexpr size_t NR = 3 * N<T,lmul>();
    constexpr size_t VL = N<T,lmul>();


    alignas(64) T dummy_a[1];
    alignas(64) T dummy_b[3 * VL];


    dummy_a[0] = static_cast<T>(1.001);

    for(size_t i = 0; i < 3 * VL; ++i)
        dummy_b[i] = static_cast<T>(1.0 + i);


    auto c0 = set0<T,lmul>();
    auto c1 = set0<T,lmul>();
    auto c2 = set0<T,lmul>();


    auto start =
        std::chrono::high_resolution_clock::now();


    for(size_t it = 0; it < iterations; ++it)
    {
        auto a = set1<T,lmul>(dummy_a[0]);


        auto b0 = load<T,lmul>(dummy_b);
        auto b1 = load<T,lmul>(dummy_b + VL);
        auto b2 = load<T,lmul>(dummy_b + 2*VL);


        c0 = fmadd(a,b0,c0);
        c1 = fmadd(a,b1,c1);
        c2 = fmadd(a,b2,c2);
    }


    auto end =
        std::chrono::high_resolution_clock::now();


    //
    // checksum
    //
    auto sum = add(c0,c1);
    sum = add(sum,c2);


    alignas(64) T result[VL];

    store(result,sum);


    double checksum = 0.0;

    for(size_t i = 0; i < VL; ++i)
        checksum += result[i];


    printf("checksum=%lf\n", checksum);


    double time =
        std::chrono::duration<double>(end-start).count();


    printf("duration=%lf\n", time);

    constexpr double flops_per_iter =
        3 * VL * 2;


    return (iterations * flops_per_iter)
        / time
        / 1e9;
}

template<typename T>
static inline
double benchmark_gemm_pattern_lmul4_k2(size_t iterations)
{
    using namespace mipp;

    constexpr int lmul = 4;

    alignas(64) T dummy_b0[N<T,lmul>()];
    alignas(64) T dummy_b1[N<T,lmul>()];

    for(size_t i = 0; i < N<T,lmul>(); ++i)
    {
        dummy_b0[i] = static_cast<T>(1.0 + i);
        dummy_b1[i] = static_cast<T>(2.0 + i);
    }

    auto c0 = set0<T,lmul>();
    auto c1 = set0<T,lmul>();
    auto c2 = set0<T,lmul>();
    auto c3 = set0<T,lmul>();

    double scalar0 = 1.001;
    double scalar1 = 2.001;

    auto start = std::chrono::high_resolution_clock::now();

    for(size_t i = 0; i < iterations; ++i)
    {
        asm volatile("" ::: "memory");

        auto a0 = set1<T,lmul>(scalar0);
        auto a1 = set1<T,lmul>(scalar1);

        auto b0 = load<T,lmul>(dummy_b0);
        auto b1 = load<T,lmul>(dummy_b1);

        asm volatile("" ::: "memory");

        c0 = fmadd(a0, b0, c0);
        c1 = fmadd(a0, b0, c1);
        c2 = fmadd(a0, b0, c2);
        c3 = fmadd(a0, b0, c3);

        c0 = fmadd(a1, b1, c0);
        c1 = fmadd(a1, b1, c1);
        c2 = fmadd(a1, b1, c2);
        c3 = fmadd(a1, b1, c3);

        scalar0 += 0.000001;
        scalar1 += 0.000002;
    }

    auto end = std::chrono::high_resolution_clock::now();

    auto sum = add(c0, c1);
    sum = add(sum, c2);
    sum = add(sum, c3);

    alignas(64) T result[N<T,lmul>()];

    store(result, sum);

    double checksum = 0.0;

    for(size_t i = 0; i < N<T,lmul>(); ++i)
        checksum += result[i];

    printf("checksum=%lf\n", checksum);

    double time =
        std::chrono::duration<double>(end - start).count();

    constexpr double flops_per_iter =
        8 * N<T,lmul>() * 2; // 8 FMAs per iteration

    printf("duration=%lf\n", time);

    return (iterations * flops_per_iter)
            / time
            / 1e9;
}

template<typename T>
static inline
double benchmark_gemm_pattern_lmul4_k2_pipeline(size_t iterations)
{
    using namespace mipp;

    constexpr int lmul = 4;

    alignas(64) T dummy_b0[8][N<T,lmul>()];
    alignas(64) T dummy_b1[8][N<T,lmul>()];

    for(size_t s = 0; s < 8; ++s)
    {
        for(size_t i = 0; i < N<T,lmul>(); ++i)
        {
            dummy_b0[s][i] = static_cast<T>(1.0 + s + i);
            dummy_b1[s][i] = static_cast<T>(2.0 + s + i);
        }
    }

    auto c0 = set0<T,lmul>();
    auto c1 = set0<T,lmul>();
    auto c2 = set0<T,lmul>();
    auto c3 = set0<T,lmul>();

    double scalar0 = 1.001;
    double scalar1 = 2.001;

    size_t idx = 0;

    //
    // Prologue
    //
    asm volatile("" ::: "memory");

    auto b0 = load<T,lmul>(dummy_b0[idx]);
    auto b1 = load<T,lmul>(dummy_b1[idx]);

    asm volatile("" ::: "memory");

    auto start = std::chrono::high_resolution_clock::now();

    for(size_t it = 0; it < iterations; ++it)
    {

        auto a0 = set1<T,lmul>(scalar0);
        auto a1 = set1<T,lmul>(scalar1);

        //
        // Compute with current operands
        //
        c0 = fmadd(a0, b0, c0);
        c1 = fmadd(a0, b0, c1);
        c2 = fmadd(a0, b0, c2);
        c3 = fmadd(a0, b0, c3);

        c0 = fmadd(a1, b1, c0);
        c1 = fmadd(a1, b1, c1);
        c2 = fmadd(a1, b1, c2);
        c3 = fmadd(a1, b1, c3);

        scalar0 += 0.000001;
        scalar1 += 0.000002;

        //
        // Preload next iteration
        //
        idx = (idx + 1) & 7;

        asm volatile("" ::: "memory");

        b0 = load<T,lmul>(dummy_b0[idx]);
        b1 = load<T,lmul>(dummy_b1[idx]);

        asm volatile("" ::: "memory");
    }

    auto end = std::chrono::high_resolution_clock::now();

    auto sum = add(c0, c1);
    sum = add(sum, c2);
    sum = add(sum, c3);

    alignas(64) T result[N<T,lmul>()];

    store(result, sum);

    double checksum = 0.0;

    for(size_t i = 0; i < N<T,lmul>(); ++i)
        checksum += result[i];

    printf("checksum=%lf\n", checksum);

    double time =
        std::chrono::duration<double>(end - start).count();

    constexpr double flops_per_iter =
        8 * N<T,lmul>() * 2;
    
    printf("duration=%lf\n", time);

    return (iterations * flops_per_iter)
            / time
            / 1e9;
}

template<typename T>
static inline
double benchmark_gemm_pattern_lmul2_k4_pipeline(size_t iterations)
{
    using namespace mipp;

    constexpr int lmul = 2;
    constexpr size_t VL = N<T,lmul>();

    alignas(64) T dummy_b[8][VL];

    for(size_t k = 0; k < 8; ++k)
    {
        for(size_t i = 0; i < VL; ++i)
        {
            dummy_b[k][i] = static_cast<T>(1.0 + k + i);
        }
    }


    auto c0 = set0<T,lmul>();
    auto c1 = set0<T,lmul>();
    auto c2 = set0<T,lmul>();
    auto c3 = set0<T,lmul>();


    double s0 = 1.001;
    double s1 = 1.002;
    double s2 = 1.003;
    double s3 = 1.004;


    size_t idx = 0;


    //
    // Prologue
    //
    auto b0 = load<T,lmul>(dummy_b[0]);
    auto b1 = load<T,lmul>(dummy_b[1]);


    auto start =
        std::chrono::high_resolution_clock::now();


    for(size_t it = 0; it < iterations; ++it)
    {
        //
        // Current broadcasts
        //
        auto a0 = set1<T,lmul>(s0);
        auto a1 = set1<T,lmul>(s1);


        //
        // First half
        //
        c0 = fmadd(a0,b0,c0);
        c1 = fmadd(a1,b1,c1);


        //
        // Software pipelined loads
        //
        idx = (idx + 2) & 7;

        auto b2 = load<T,lmul>(dummy_b[idx]);
        auto b3 = load<T,lmul>(dummy_b[(idx+1)&7]);


        //
        // Second half
        //
        auto a2 = set1<T,lmul>(s2);
        auto a3 = set1<T,lmul>(s3);

        c2 = fmadd(a2,b2,c2);
        c3 = fmadd(a3,b3,c3);


        //
        // Rotate for next iteration
        //
        b0 = b2;
        b1 = b3;


        s0 += 0.000001;
        s1 += 0.000002;
        s2 += 0.000003;
        s3 += 0.000004;
    }


    auto end =
        std::chrono::high_resolution_clock::now();


    auto sum = add(c0,c1);
    sum = add(sum,c2);
    sum = add(sum,c3);


    alignas(64) T result[VL];

    store(result,sum);


    double checksum = 0.0;

    for(size_t i = 0; i < VL; ++i)
        checksum += result[i];


    printf("checksum=%lf\n", checksum);


    double time =
        std::chrono::duration<double>(end-start).count();


    //
    // 4 FMAs per iteration
    //
    constexpr double flops_per_iter =
        4 * VL * 2;

    printf("duration=%lf\n", time);
    return (iterations * flops_per_iter)
            / time
            / 1e9;
}

template<typename T>
static inline
double benchmark_dot_gemm_lmul4(size_t iterations)
{
    using namespace mipp;

    constexpr int lmul = 4;
    constexpr size_t VL = N<T,lmul>();

    alignas(64) T dummy_a[4][VL];
    alignas(64) T dummy_b[4][VL];

    for(size_t j = 0; j < 4; ++j)
    {
        for(size_t i = 0; i < VL; ++i)
        {
            dummy_a[j][i] = static_cast<T>(1.0 + i + j);
            dummy_b[j][i] = static_cast<T>(2.0 + i + j);
        }
    }


    auto c0 = set0<T,lmul>();
    auto c1 = set0<T,lmul>();
    auto c2 = set0<T,lmul>();
    auto c3 = set0<T,lmul>();


    auto start =
        std::chrono::high_resolution_clock::now();


    for(size_t it = 0; it < iterations; ++it)
    {
        auto a0 = load<T,lmul>(dummy_a[0]);
        auto b0 = load<T,lmul>(dummy_b[0]);

        auto a1 = load<T,lmul>(dummy_a[1]);
        auto b1 = load<T,lmul>(dummy_b[1]);

        auto a2 = load<T,lmul>(dummy_a[2]);
        auto b2 = load<T,lmul>(dummy_b[2]);

        auto a3 = load<T,lmul>(dummy_a[3]);
        auto b3 = load<T,lmul>(dummy_b[3]);


        c0 = fmadd(a0,b0,c0);
        c1 = fmadd(a1,b1,c1);
        c2 = fmadd(a2,b2,c2);
        c3 = fmadd(a3,b3,c3);
    }


    auto end =
        std::chrono::high_resolution_clock::now();


    //
    // Reduction
    //
    T r0 = hadd(c0);
    T r1 = hadd(c1);
    T r2 = hadd(c2);
    T r3 = hadd(c3);


    double checksum =
        r0+r1+r2+r3;


    printf("checksum=%lf\n", checksum);


    double time =
        std::chrono::duration<double>(end-start).count();


    //
    // 4 dot products per iteration
    //
    constexpr double flops_per_iter =
        4 * VL * 2;

    printf("duration=%lf\n", time);

    return (iterations * flops_per_iter)
            / time
            / 1e9;
}

template<typename T>
static inline
double benchmark_dot_gemm_lmul4_2dot(size_t iterations)
{
    using namespace mipp;

    constexpr int lmul = 4;
    constexpr size_t VL = N<T,lmul>();

    alignas(64) T dummy_a[2][VL];
    alignas(64) T dummy_b[2][VL];

    for(size_t j = 0; j < 2; ++j)
    {
        for(size_t i = 0; i < VL; ++i)
        {
            dummy_a[j][i] = static_cast<T>(1.0 + j + i);
            dummy_b[j][i] = static_cast<T>(2.0 + j + i);
        }
    }


    auto c0 = set0<T,lmul>();
    auto c1 = set0<T,lmul>();
    volatile size_t idx = 0;


    auto start =
        std::chrono::high_resolution_clock::now();


    for(size_t it = 0; it < iterations; ++it)
    {
        size_t idx = it & 1;
        size_t idx2 = (it+1) & 1;

        auto a0 = load<T,lmul>(dummy_a[idx]);
        auto b0 = load<T,lmul>(dummy_b[idx]);

        auto a1 = load<T,lmul>(dummy_a[idx2]);
        auto b1 = load<T,lmul>(dummy_b[idx2]);

        c0 = fmadd(a0,b0,c0);
        c1 = fmadd(a1,b1,c1);
    }


    auto end = std::chrono::high_resolution_clock::now();


    auto r0 = hadd(c0);
    auto r1 = hadd(c1);


    printf("checksum=%lf\n",
           static_cast<double>(r0+r1));


    double time = std::chrono::duration<double>(end-start).count();


    constexpr double flops_per_iter = 2 * VL * 2;


    printf("duration=%lf\n", time);

    return (iterations * flops_per_iter)
            / time
            / 1e9;
}

int main(int argc, char** argv)
{
    size_t iterations = std::stoull(argv[1]);

    double gflops = 0;
    printf("pure FMA\n");
    printf("%.3f GFLOP/s\n\n",
        run_pure_fma(iterations));


    printf("broadcast + FMA\n");
    printf("%.3f GFLOP/s\n\n",
        run_broadcast_fma(iterations));


    printf("load + FMA\n");
    gflops = benchmark_load_fma<T>(iterations);
    printf("%.3f GFLOP/s\n\n", gflops);


    printf("load + FMA LMUL=4\n");
    gflops = benchmark_load_fma_lmul4<T>(iterations);
    printf("%.3f GFLOP/s\n\n", gflops);

    printf("load + FMA LMUL=4 2 loads\n");
    gflops = benchmark_load_fma_lmul4_2loads<T>(iterations);
    printf("%.3f GFLOP/s\n\n", gflops);

    printf("load + FMA LMUL=4 3 loads\n");
    gflops = benchmark_load_fma_lmul4_3loads<T>(iterations);
    printf("%.3f GFLOP/s\n\n", gflops);

    printf("broadcast + FMA LMUL=4\n");
    gflops = benchmark_fma_lmul4_broadcasts<T>(iterations);
    printf("%.3f GFLOP/s\n\n", gflops);

    printf("broadcast + FMA LMUL=4 2 broadcasts\n");
    gflops = benchmark_fma_lmul4_2broadcasts<T>(iterations);
    printf("%.3f GFLOP/s\n\n", gflops);

    printf("GEMM pattern LMUL=4\n");
    gflops = benchmark_gemm_pattern_lmul4<T>(iterations);
    printf("%.3f GFLOP/s\n\n", gflops);

    printf("GEMM pattern LMUL=4 MR=1 NR=6\n");
    gflops = benchmark_gemm_pattern_lmul4_mr1_nr6<T>(iterations);
    printf("%.3f GFLOP/s\n\n", gflops);

    printf("GEMM pattern LMUL=4 MR=1 NR=5\n");
    gflops = benchmark_gemm_pattern_lmul4_mr1_nr5<T>(iterations);
    printf("%.3f GFLOP/s\n\n", gflops);

    printf("GEMM pattern LMUL=8\n");
    gflops = benchmark_gemm_pattern_lmul8<T>(iterations);
    printf("%.3f GFLOP/s\n\n", gflops);

    printf("GEMM pattern LMUL=8 NR=2\n");
    gflops = benchmark_gemm_pattern_lmul8_nr2<T>(iterations);
    printf("%.3f GFLOP/s\n\n", gflops);

    printf("GEMM pattern LMUL=4 NR=3\n");
    gflops = benchmark_gemm_pattern_lmul4_nr3<T>(iterations);
    printf("%.3f GFLOP/s\n\n", gflops);

    printf("GEMM pattern LMUL=4 K=2\n");
    gflops = benchmark_gemm_pattern_lmul4_k2<T>(iterations);
    printf("%.3f GFLOP/s\n\n", gflops);

    printf("GEMM pattern LMUL=4 K=2 pipeline\n");
    gflops = benchmark_gemm_pattern_lmul4_k2_pipeline<T>(iterations);
    printf("%.3f GFLOP/s\n\n", gflops);

    printf("GEMM pattern LMUL=2 K=4 pipeline\n");
    gflops = benchmark_gemm_pattern_lmul2_k4_pipeline<T>(iterations);
    printf("%.3f GFLOP/s\n\n", gflops);

    printf("Dot GEMM LMUL=4\n");
    gflops = benchmark_dot_gemm_lmul4<T>(iterations);
    printf("%.3f GFLOP/s\n\n", gflops);

    printf("Dot GEMM LMUL=4 2 dot\n");
    gflops = benchmark_dot_gemm_lmul4_2dot<T>(iterations);
    printf("%.3f GFLOP/s\n\n", gflops); 

}