/*
 * - generated with asmgen-gemm
 * - Assuming vle and vfmacc share pipelines, theo. peak performance is 7.467 FLOP/cycle
 * - tuned with x60_multiloopexplore.py (FLOP/cycle reached: 7.32)
 * - 2vx15 kernel could potentially have 7.5 FLOP/cycle but during tuning 7.245 was not exceeded
 * - beta*c/alpha*ab/store written manually
 */

#include "blis.h"

#define SIZESHIFT "2"
#define SIZEBITS "32"

void bli_sgemm_x60_2vx14_2u(dim_t m, dim_t n, dim_t k, 
                            const void* alpha,
                            const void* a,
                            const void* b,
                            const void* beta,
                            void* c, inc_t rs_c0, inc_t cs_c0,
                            const auxinfo_t* data, const cntx_t*    cntx)
{
    // unused for now, but keep for later memory/cache tuning
    const void* a_next = bli_auxinfo_next_a( data );
    const void* b_next = bli_auxinfo_next_b( data );

    //const uint64_t unroll = 2;

    //uint64_t kc = k / unroll;
    //uint64_t kleft = k % unroll;
    int64_t rs_c   = rs_c0;
    int64_t cs_c   = cs_c0;

    //printf("rs_c: %llu\n",rs_c);
    //printf("cs_c: %llu\n",cs_c);

    // vlen should be half of MR
    uint64_t vlen = bli_cntx_get_blksz_def_dt( BLIS_SINGLE, BLIS_MR, cntx )/2;

    vlen *= sizeof(float);

    // override vlen
    __asm__(
            //"csrr %[vlen],vlenb\n\t"
            "vsetvli %[vlen], %[vlen], e8, m1, ta, ma\n\t"
            : [vlen] "+r" (vlen)
            :
            :
       );

    vlen = vlen/sizeof(float);

    //printf("GEMM with m=%lld,n=%lld,k=%lld, rs_c=%lld, cs_c=%lld\n",
    //        m, n, k, rs_c, cs_c);

    GEMM_UKR_SETUP_CT( s, vlen*2, 14, false );


    // prefetch distance
    uint64_t pfdist = 7;
    uint64_t vlenx2;
    uint64_t unroll;
    uint64_t counter;

    uint64_t aptr2;
    uint64_t bptr2;
    uint64_t cptr2;

    __asm__ (
        // scalars
        //"ld %[cs_c], 48(s2)\n\t"  // alpha
        //"ld s5, 56(s2)\n\t" // beta
        "flw f0, 0(%[alpha])\n\t"
        "flw f1, 0(%[beta])\n\t"
        
        //"add s2, %[inputs], 0\n\t"
        //"ld %[cs_c], 32(s2)\n\t" // rs_c
        "add %[rs_c], %[rs_c], -1\n\t"
        "beq %[rs_c], zero, .rscokay%=\n\t"
        "unimp\n\t" // Fail if rs_c != 1
        ".rscokay%=:"
        //"ld %[a_next], 88(s2)\n\t" // a_next
        //"ld %[a_next], 96(s2)\n\t" // b_next
        //"ld %[vlen], 24(s2)\n\t" // vlen
        "vsetvli %[vlen], %[vlen], e" SIZEBITS ", m1, ta, ma\n\t"
        "slli %[vlen],%[vlen]," SIZESHIFT "\n\t"
        "slli %[vlenx2],%[vlen],1\n\t"

        //unroll
        "li %[unroll], 2\n\t"
        "divu %[counter], %[n], %[unroll]\n\t"
        "remu %[n], %[n], %[unroll]\n\t"

        
        // C-tile
        //"ld %[cptr], 80(s2)\n\t" // c
        "add %[cptr2], %[cptr], 0\n\t"
        //"ld %[cs_c], 40(s2)\n\t" // cs_c
        "slli %[cs_c], %[cs_c], " SIZESHIFT "\n\t"
        // counters
        //"ld %[counter], 8(s2)\n\t" // kc
        //"ld s5, 16(s2)\n\t" // kleft
        // pointers
        //"ld %[bptr], 72(s2)\n\t" // b
        "add %[bptr2], %[bptr], 0\n\t"
        //"ld %[aptr], 64(s2)\n\t" // a
        "add %[aptr2], %[aptr], %[vlen]\n\t"


        // preload/zeroing
        "vmv.v.i v1,0\n\t"
        "vmv.v.i v2,0\n\t"
        "vle32.v v29, (%[aptr])\n\t"
        "flw f2, 0(%[bptr])\n\t"
        "vmv.v.i v3,0\n\t"
        "vmv.v.i v4,0\n\t"
        "flw f3, 4(%[bptr])\n\t"
        "vmv.v.i v5,0\n\t"
        "vmv.v.i v6,0\n\t"
        "flw f4, 8(%[bptr])\n\t"
        "vmv.v.i v7,0\n\t"
        "vmv.v.i v8,0\n\t"
        "flw f5, 12(%[bptr])\n\t"
        "vle32.v v30, (%[aptr2])\n\t"
        "vmv.v.i v9,0\n\t"
        "vmv.v.i v10,0\n\t"
        "flw f6, 16(%[bptr])\n\t"
        "vmv.v.i v11,0\n\t"
        "vmv.v.i v12,0\n\t"
        "flw f7, 20(%[bptr])\n\t"
        "vmv.v.i v13,0\n\t"
        "vmv.v.i v14,0\n\t"
        "flw f8, 24(%[bptr])\n\t"
        "vmv.v.i v15,0\n\t"
        "vmv.v.i v16,0\n\t"
        "flw f9, 28(%[bptr])\n\t"
        "vmv.v.i v17,0\n\t"
        "flw f10, 32(%[bptr])\n\t"
        "vmv.v.i v18,0\n\t"
        "vmv.v.i v19,0\n\t"
        "flw f11, 36(%[bptr])\n\t"
        "vmv.v.i v20,0\n\t"
        "flw f12, 40(%[bptr])\n\t"
        "vmv.v.i v21,0\n\t"
        "vmv.v.i v22,0\n\t"
        "flw f13, 44(%[bptr])\n\t"
        "vmv.v.i v23,0\n\t"
        "flw f14, 48(%[bptr])\n\t"
        "vmv.v.i v24,0\n\t"
        "vmv.v.i v25,0\n\t"
        "flw f15, 52(%[bptr])\n\t"
        "vmv.v.i v26,0\n\t"
        "vmv.v.i v27,0\n\t"
        "vmv.v.i v28,0\n\t"
        "beq %[counter], zero, .kdone%=\n\t"
        "add %[aptr], %[aptr], %[vlenx2]\n\t"
        "add %[aptr2], %[aptr2], %[vlenx2]\n\t"
        "add %[bptr], %[bptr], 56\n\t"
        "add %[counter],%[counter],-1\n\t"
        "beq %[counter], zero, .klast%=\n\t"
        "j .kloop%=\n\t"
        ".prefetchc%=:\n\t"
        "prefetch.w 0(%[cptr2])\n\t"
        "add %[cptr2], %[cptr2], %[cs_c]\n\t"
        "prefetch.w 0(%[cptr2])\n\t"
        "add %[cptr2], %[cptr2], %[cs_c]\n\t"
        "prefetch.w 0(%[cptr2])\n\t"
        "add %[cptr2], %[cptr2], %[cs_c]\n\t"
        "prefetch.w 0(%[cptr2])\n\t"
        "add %[cptr2], %[cptr2], %[cs_c]\n\t"
        "prefetch.w 0(%[cptr2])\n\t"
        "add %[cptr2], %[cptr2], %[cs_c]\n\t"
        "prefetch.w 0(%[cptr2])\n\t"
        "add %[cptr2], %[cptr2], %[cs_c]\n\t"
        "prefetch.w 0(%[cptr2])\n\t"
        "add %[cptr2], %[cptr2], %[cs_c]\n\t"
        "prefetch.w 0(%[cptr2])\n\t"
        "add %[cptr2], %[cptr2], %[cs_c]\n\t"
        "prefetch.w 0(%[cptr2])\n\t"
        "add %[cptr2], %[cptr2], %[cs_c]\n\t"
        "prefetch.w 0(%[cptr2])\n\t"
        "add %[cptr2], %[cptr2], %[cs_c]\n\t"
        "prefetch.w 0(%[cptr2])\n\t"
        "add %[cptr2], %[cptr2], %[cs_c]\n\t"
        "prefetch.w 0(%[cptr2])\n\t"
        "add %[cptr2], %[cptr2], %[cs_c]\n\t"
        "prefetch.w 0(%[cptr2])\n\t"
        "add %[cptr2], %[cptr2], %[cs_c]\n\t"
        "prefetch.w 0(%[cptr2])\n\t"
        ".kloop%=:\n\t"
            //"ble %[counter], %[pfdist], .skipbpref%=\n\t"
            //"prefetch.r 768(%[aptr])\n\t"
            //"prefetch.r 1344(%[bptr])\n\t"
            //".skipbpref%=:"

            "vfmacc.vf v1, f2, v29\n\t"
            "vfmacc.vf v2, f2, v30\n\t"
            "vfmacc.vf v3, f3, v29\n\t"
            "flw f19, 12(%[bptr])\n\t"
            "vle32.v v31, (%[aptr])\n\t"
            "vfmacc.vf v4, f3, v30\n\t"
            "vle32.v v0, (%[aptr2])\n\t"
            "vfmacc.vf v5, f4, v29\n\t"
            "vfmacc.vf v6, f4, v30\n\t"
            "vfmacc.vf v7, f5, v29\n\t"
            "flw f18, 8(%[bptr])\n\t"
            "vfmacc.vf v8, f5, v30\n\t"
            "vfmacc.vf v9, f6, v29\n\t"
            "flw f16, 0(%[bptr])\n\t"
            "vfmacc.vf v10, f6, v30\n\t"
            "vfmacc.vf v11, f7, v29\n\t"
            "flw f20, 16(%[bptr])\n\t"
            "vfmacc.vf v12, f7, v30\n\t"
            "flw f21, 20(%[bptr])\n\t"
            "vfmacc.vf v13, f8, v29\n\t"
            "add %[aptr], %[aptr], %[vlenx2]\n\t"
            "vfmacc.vf v14, f8, v30\n\t"
            "flw f22, 24(%[bptr])\n\t"
            "vfmacc.vf v15, f9, v29\n\t"
            "flw f17, 4(%[bptr])\n\t"
            "add %[aptr2], %[aptr2], %[vlenx2]\n\t"
            "vfmacc.vf v16, f9, v30\n\t"
            "vfmacc.vf v17, f10, v29\n\t"
            "flw f23, 28(%[bptr])\n\t"
            "vfmacc.vf v18, f10, v30\n\t"
            "flw f24, 32(%[bptr])\n\t"
            "vfmacc.vf v19, f11, v29\n\t"
            "vfmacc.vf v20, f11, v30\n\t"
            "flw f25, 36(%[bptr])\n\t"
            "add %[bptr2], %[bptr2], 112\n\t"
            "vfmacc.vf v21, f12, v29\n\t"
            "vfmacc.vf v22, f12, v30\n\t"
            "flw f26, 40(%[bptr])\n\t"
            "vfmacc.vf v23, f13, v29\n\t"
            "flw f9, 28(%[bptr2])\n\t"
            "vfmacc.vf v24, f13, v30\n\t"
            "flw f27, 44(%[bptr])\n\t"
            "vfmacc.vf v25, f14, v29\n\t"
            "nop\n\t"
            "vfmacc.vf v26, f14, v30\n\t"
            "flw f28, 48(%[bptr])\n\t"
            "vfmacc.vf v27, f15, v29\n\t"
            "vfmacc.vf v28, f15, v30\n\t"
            "flw f29, 52(%[bptr])\n\t"
            "nop\n\t"
            "vfmacc.vf v1, f16, v31\n\t"
            "flw f3, 4(%[bptr2])\n\t"
            "vfmacc.vf v2, f16, v0\n\t"
            "flw f4, 8(%[bptr2])\n\t"
            "add %[bptr], %[bptr], 112\n\t"
            "vfmacc.vf v3, f17, v31\n\t"
            "vfmacc.vf v4, f17, v0\n\t"
            "vfmacc.vf v5, f18, v31\n\t"
            "vfmacc.vf v6, f18, v0\n\t"
            "vfmacc.vf v7, f19, v31\n\t"
            "flw f5, 12(%[bptr2])\n\t"
            "vle32.v v29, (%[aptr])\n\t"
            "vfmacc.vf v8, f19, v0\n\t"
            "vfmacc.vf v9, f20, v31\n\t"
            "vfmacc.vf v10, f20, v0\n\t"
            "flw f6, 16(%[bptr2])\n\t"
            "vfmacc.vf v11, f21, v31\n\t"
            "vfmacc.vf v12, f21, v0\n\t"
            "flw f7, 20(%[bptr2])\n\t"
            "vfmacc.vf v13, f22, v31\n\t"
            "vfmacc.vf v14, f22, v0\n\t"
            "flw f2, 0(%[bptr2])\n\t"
            "nop\n\t"
            "vle32.v v30, (%[aptr2])\n\t"
            "vfmacc.vf v15, f23, v31\n\t"
            "vfmacc.vf v16, f23, v0\n\t"
            "flw f10, 32(%[bptr2])\n\t"
            "add %[counter],%[counter],-1\n\t"
            "vfmacc.vf v17, f24, v31\n\t"
            "vfmacc.vf v18, f24, v0\n\t"
            "vfmacc.vf v19, f25, v31\n\t"
            "vfmacc.vf v20, f25, v0\n\t"
            "flw f11, 36(%[bptr2])\n\t"
            "vfmacc.vf v21, f26, v31\n\t"
            "add %[aptr2], %[aptr2], %[vlenx2]\n\t"
            "vfmacc.vf v22, f26, v0\n\t"
            "flw f12, 40(%[bptr2])\n\t"
            "vfmacc.vf v23, f27, v31\n\t"
            "vfmacc.vf v24, f27, v0\n\t"
            "flw f13, 44(%[bptr2])\n\t"
            "vfmacc.vf v25, f28, v31\n\t"
            "flw f8, 24(%[bptr2])\n\t"
            "add %[aptr], %[aptr], %[vlenx2]\n\t"
            "vfmacc.vf v26, f28, v0\n\t"
            "flw f14, 48(%[bptr2])\n\t"
            "vfmacc.vf v27, f29, v31\n\t"
            "vfmacc.vf v28, f29, v0\n\t"
            "flw f15, 52(%[bptr2])\n\t"
            "beq %[counter], %[pfdist], .prefetchc%=\n\t"
            "bnez %[counter], .kloop%=\n\t"
        ".klast%=:\n\t"
            "vfmacc.vf v1, f2, v29\n\t"
            "vfmacc.vf v2, f2, v30\n\t"
            "vfmacc.vf v3, f3, v29\n\t"
            "vfmacc.vf v4, f3, v30\n\t"
            "vle32.v v31, (%[aptr])\n\t"
            "vle32.v v0, (%[aptr2])\n\t"
            "vfmacc.vf v5, f4, v29\n\t"
            "vfmacc.vf v6, f4, v30\n\t"
            "flw f19, 12(%[bptr])\n\t"
            "vfmacc.vf v7, f5, v29\n\t"
            "flw f18, 8(%[bptr])\n\t"
            "vfmacc.vf v8, f5, v30\n\t"
            "vfmacc.vf v9, f6, v29\n\t"
            "flw f16, 0(%[bptr])\n\t"
            "vfmacc.vf v10, f6, v30\n\t"
            "vfmacc.vf v11, f7, v29\n\t"
            "flw f20, 16(%[bptr])\n\t"
            "vfmacc.vf v12, f7, v30\n\t"
            "flw f21, 20(%[bptr])\n\t"
            "vfmacc.vf v13, f8, v29\n\t"
            "add %[aptr], %[aptr], %[vlenx2]\n\t"
            "vfmacc.vf v14, f8, v30\n\t"
            "flw f22, 24(%[bptr])\n\t"
            "vfmacc.vf v15, f9, v29\n\t"
            "flw f17, 4(%[bptr])\n\t"
            "add %[aptr2], %[aptr2], %[vlenx2]\n\t"
            "vfmacc.vf v16, f9, v30\n\t"
            "flw f23, 28(%[bptr])\n\t"
            "vfmacc.vf v17, f10, v29\n\t"
            "vfmacc.vf v18, f10, v30\n\t"
            "flw f24, 32(%[bptr])\n\t"
            "vfmacc.vf v19, f11, v29\n\t"
            "vfmacc.vf v20, f11, v30\n\t"
            "flw f25, 36(%[bptr])\n\t"
            "add %[bptr2], %[bptr2], 112\n\t"
            "vfmacc.vf v21, f12, v29\n\t"
            "vfmacc.vf v22, f12, v30\n\t"
            "flw f26, 40(%[bptr])\n\t"
            "vfmacc.vf v23, f13, v29\n\t"
            "vfmacc.vf v24, f13, v30\n\t"
            "flw f27, 44(%[bptr])\n\t"
            "vfmacc.vf v25, f14, v29\n\t"
            "nop\n\t"
            "vfmacc.vf v26, f14, v30\n\t"
            "flw f28, 48(%[bptr])\n\t"
            "vfmacc.vf v27, f15, v29\n\t"
            "vfmacc.vf v28, f15, v30\n\t"
            "flw f29, 52(%[bptr])\n\t"
            "nop\n\t"
            "vfmacc.vf v1, f16, v31\n\t"
            "vfmacc.vf v2, f16, v0\n\t"
            "vfmacc.vf v3, f17, v31\n\t"
            "vfmacc.vf v4, f17, v0\n\t"
            "vfmacc.vf v5, f18, v31\n\t"
            "vfmacc.vf v6, f18, v0\n\t"
            "vfmacc.vf v7, f19, v31\n\t"
            "vfmacc.vf v8, f19, v0\n\t"
            "vfmacc.vf v9, f20, v31\n\t"
            "vfmacc.vf v10, f20, v0\n\t"
            "vfmacc.vf v11, f21, v31\n\t"
            "vfmacc.vf v12, f21, v0\n\t"
            "vfmacc.vf v13, f22, v31\n\t"
            "vfmacc.vf v14, f22, v0\n\t"
            "nop\n\t"
            "vfmacc.vf v15, f23, v31\n\t"
            "vfmacc.vf v16, f23, v0\n\t"
            "vfmacc.vf v17, f24, v31\n\t"
            "vfmacc.vf v18, f24, v0\n\t"
            "vfmacc.vf v19, f25, v31\n\t"
            "vfmacc.vf v20, f25, v0\n\t"
            "vfmacc.vf v21, f26, v31\n\t"
            "vfmacc.vf v22, f26, v0\n\t"
            "vfmacc.vf v23, f27, v31\n\t"
            "vfmacc.vf v24, f27, v0\n\t"
            "vfmacc.vf v25, f28, v31\n\t"
            "vfmacc.vf v26, f28, v0\n\t"
            "vfmacc.vf v27, f29, v31\n\t"
            "vfmacc.vf v28, f29, v0\n\t"
//            "add %[counter],%[counter],-1\n\t"
        ".kdone%=:\n\t"
        // move rest into counter
        "mv %[counter], %[n]\n\t"
        "beq %[counter], zero, .k1done%=\n\t"
        ".k1loop%=:\n\t"
            "vle32.v v29, (%[aptr])\n\t"
            "vle32.v v30, (%[aptr2])\n\t"
            "flw f2, 0(%[bptr2])\n\t"
            "flw f3, 4(%[bptr2])\n\t"
            "flw f4, 8(%[bptr2])\n\t"
            "vfmacc.vf v1, f2, v29\n\t"
            "vfmacc.vf v2, f2, v30\n\t"
            "flw f5, 12(%[bptr2])\n\t"
            "vfmacc.vf v3, f3, v29\n\t"
            "vfmacc.vf v4, f3, v30\n\t"
            "flw f6, 16(%[bptr2])\n\t"
            "vfmacc.vf v5, f4, v29\n\t"
            "vfmacc.vf v6, f4, v30\n\t"
            "flw f7, 20(%[bptr2])\n\t"
            "vfmacc.vf v7, f5, v29\n\t"
            "vfmacc.vf v8, f5, v30\n\t"
            "flw f8, 24(%[bptr2])\n\t"
            "vfmacc.vf v9, f6, v29\n\t"
            "add %[aptr], %[aptr], %[vlenx2]\n\t"
            "vfmacc.vf v10, f6, v30\n\t"
            "flw f9, 28(%[bptr2])\n\t"
            "vfmacc.vf v11, f7, v29\n\t"
            "add %[aptr2], %[aptr2], %[vlenx2]\n\t"
            "vfmacc.vf v12, f7, v30\n\t"
            "flw f10, 32(%[bptr2])\n\t"
            "vfmacc.vf v13, f8, v29\n\t"
            "vfmacc.vf v14, f8, v30\n\t"
            "flw f11, 36(%[bptr2])\n\t"
            "vfmacc.vf v15, f9, v29\n\t"
            "vfmacc.vf v16, f9, v30\n\t"
            "flw f12, 40(%[bptr2])\n\t"
            "vfmacc.vf v17, f10, v29\n\t"
            "vfmacc.vf v18, f10, v30\n\t"
            "flw f13, 44(%[bptr2])\n\t"
            "vfmacc.vf v19, f11, v29\n\t"
            "vfmacc.vf v20, f11, v30\n\t"
            "flw f14, 48(%[bptr2])\n\t"
            "vfmacc.vf v21, f12, v29\n\t"
            "add %[counter],%[counter],-1\n\t"
            "vfmacc.vf v22, f12, v30\n\t"
            "flw f15, 52(%[bptr2])\n\t"
            "vfmacc.vf v23, f13, v29\n\t"
            "vfmacc.vf v24, f13, v30\n\t"
            "add %[bptr2], %[bptr2], 56\n\t"
            "vfmacc.vf v25, f14, v29\n\t"
            "vfmacc.vf v26, f14, v30\n\t"
            "vfmacc.vf v27, f15, v29\n\t"
            "vfmacc.vf v28, f15, v30\n\t"
            "bnez %[counter], .k1loop%=\n\t"
        ".k1done%=:\n\t"
        "li %[bptr], 1\n\t"
        "fcvt.s.l f4, %[bptr]\n\t" // f4=1.0
        "fmv.s.x f5, zero\n\t" // f5=0.0
        "feq.s %[aptr], f0, f4\n\t"
        "bnez %[aptr],.alphaone%=\n\t"
        ".alphascale%=:\n\t"
            "vfmul.vf v1, v1, f0\n\t"
            "vfmul.vf v2, v2, f0\n\t"
            "vfmul.vf v3, v3, f0\n\t"
            "vfmul.vf v4, v4, f0\n\t"
            "vfmul.vf v5, v5, f0\n\t"
            "vfmul.vf v6, v6, f0\n\t"
            "vfmul.vf v7, v7, f0\n\t"
            "vfmul.vf v8, v8, f0\n\t"
            "vfmul.vf v9, v9, f0\n\t"
            "vfmul.vf v10, v10, f0\n\t"
            "vfmul.vf v11, v11, f0\n\t"
            "vfmul.vf v12, v12, f0\n\t"
            "vfmul.vf v13, v13, f0\n\t"
            "vfmul.vf v14, v14, f0\n\t"
            "vfmul.vf v15, v15, f0\n\t"
            "vfmul.vf v16, v16, f0\n\t"
            "vfmul.vf v17, v17, f0\n\t"
            "vfmul.vf v18, v18, f0\n\t"
            "vfmul.vf v19, v19, f0\n\t"
            "vfmul.vf v20, v20, f0\n\t"
            "vfmul.vf v21, v21, f0\n\t"
            "vfmul.vf v22, v22, f0\n\t"
            "vfmul.vf v23, v23, f0\n\t"
            "vfmul.vf v24, v24, f0\n\t"
            "vfmul.vf v25, v25, f0\n\t"
            "vfmul.vf v26, v26, f0\n\t"
            "vfmul.vf v27, v27, f0\n\t"
            "vfmul.vf v28, v28, f0\n\t"
        ".alphaone%=:\n\t"


        "add %[aptr], %[cptr], 0\n\t"   // load: c
        "feq.s %[counter], f1, f5\n\t" // >---------------------------
        "add %[aptr2], %[aptr], %[vlen]\n\t"      // load: c+vlen            |
        "add %[bptr2], %[aptr], 0\n\t"       // store: c                |
        "add %[cptr], %[aptr2], 0\n\t"       // store: c+vlen           |
        "bnez %[counter], .betazero%=\n\t" // <-----------------------
        ".betascale%=:\n\t" // TODO: written manually and unoptimized
            
            "vle32.v v29, (%[aptr])\n\t"  // c0_1
            "vle32.v v30, (%[aptr2])\n\t"  // c0_2
            "add %[aptr], %[aptr], %[cs_c]\n\t"
            "add %[aptr2], %[aptr2], %[cs_c]\n\t"
            "vle32.v v31, (%[aptr])\n\t"  // c1_1
            "vle32.v v0, (%[aptr2])\n\t"  // c1_2
            "vfmacc.vf v1,  f1, v29\n\t" // c0_1
            "vfmacc.vf v2,  f1, v30\n\t" // c0_2

            "add %[aptr], %[aptr], %[cs_c]\n\t"
            "add %[aptr2], %[aptr2], %[cs_c]\n\t"
            "vle32.v v29, (%[aptr])\n\t" // c2_1
            "vle32.v v30, (%[aptr2])\n\t" // c2_2
            "vfmacc.vf v3,  f1, v31\n\t"
            "vfmacc.vf v4,  f1, v0\n\t"

            "vse32.v v1, (%[bptr2])\n\t" // c0_1
            "vse32.v v2, (%[cptr])\n\t" // c0_2
            "add %[bptr2], %[bptr2], %[cs_c]\n\t"
            "add %[cptr], %[cptr], %[cs_c]\n\t"

            "add %[aptr], %[aptr], %[cs_c]\n\t"
            "add %[aptr2], %[aptr2], %[cs_c]\n\t"
            "vle32.v v31, (%[aptr])\n\t"
            "vle32.v v0, (%[aptr2])\n\t"
            "vse32.v v3, (%[bptr2])\n\t"
            "vse32.v v4, (%[cptr])\n\t"
            "add %[bptr2], %[bptr2], %[cs_c]\n\t"
            "add %[cptr], %[cptr], %[cs_c]\n\t"
            "vfmacc.vf v5,  f1, v29\n\t"
            "vfmacc.vf v6,  f1, v30\n\t"

            "add %[aptr], %[aptr], %[cs_c]\n\t"
            "add %[aptr2], %[aptr2], %[cs_c]\n\t"
            "vse32.v v5, (%[bptr2])\n\t"
            "vse32.v v6, (%[cptr])\n\t"
            "vle32.v v1, (%[aptr])\n\t"
            "vle32.v v2, (%[aptr2])\n\t"
            "add %[bptr2], %[bptr2], %[cs_c]\n\t"
            "add %[cptr], %[cptr], %[cs_c]\n\t"
            "vfmacc.vf v7,  f1, v31\n\t"
            "vfmacc.vf v8,  f1, v0\n\t"

            "add %[aptr], %[aptr], %[cs_c]\n\t"
            "add %[aptr2], %[aptr2], %[cs_c]\n\t"
            "vse32.v v7, (%[bptr2])\n\t"
            "vse32.v v8, (%[cptr])\n\t"
            "vle32.v v3, (%[aptr])\n\t"
            "vle32.v v4, (%[aptr2])\n\t"
            "add %[bptr2], %[bptr2], %[cs_c]\n\t"
            "add %[cptr], %[cptr], %[cs_c]\n\t"
            "vfmacc.vf v9,  f1, v1\n\t"
            "vfmacc.vf v10,  f1, v2\n\t"

            "add %[aptr], %[aptr], %[cs_c]\n\t"
            "add %[aptr2], %[aptr2], %[cs_c]\n\t"
            "vse32.v v9, (%[bptr2])\n\t"
            "vse32.v v10, (%[cptr])\n\t"
            "vle32.v v5, (%[aptr])\n\t"
            "vle32.v v6, (%[aptr2])\n\t"
            "add %[bptr2], %[bptr2], %[cs_c]\n\t"
            "add %[cptr], %[cptr], %[cs_c]\n\t"
            "vfmacc.vf v11,  f1, v3\n\t"
            "vfmacc.vf v12,  f1, v4\n\t"

            "add %[aptr], %[aptr], %[cs_c]\n\t"
            "add %[aptr2], %[aptr2], %[cs_c]\n\t"
            "vse32.v v11, (%[bptr2])\n\t"
            "vse32.v v12, (%[cptr])\n\t"
            "vle32.v v7, (%[aptr])\n\t"
            "vle32.v v8, (%[aptr2])\n\t"
            "add %[bptr2], %[bptr2], %[cs_c]\n\t"
            "add %[cptr], %[cptr], %[cs_c]\n\t"
            "vfmacc.vf v13,  f1, v5\n\t"
            "vfmacc.vf v14,  f1, v6\n\t"

            "add %[aptr], %[aptr], %[cs_c]\n\t"
            "add %[aptr2], %[aptr2], %[cs_c]\n\t"
            "vse32.v v13, (%[bptr2])\n\t"
            "vse32.v v14, (%[cptr])\n\t"
            "vle32.v v9, (%[aptr])\n\t"
            "vle32.v v10, (%[aptr2])\n\t"
            "add %[bptr2], %[bptr2], %[cs_c]\n\t"
            "add %[cptr], %[cptr], %[cs_c]\n\t"
            "vfmacc.vf v15,  f1, v7\n\t"
            "vfmacc.vf v16,  f1, v8\n\t"

            "add %[aptr], %[aptr], %[cs_c]\n\t"
            "add %[aptr2], %[aptr2], %[cs_c]\n\t"
            "vse32.v v15, (%[bptr2])\n\t"
            "vse32.v v16, (%[cptr])\n\t"
            "vle32.v v11, (%[aptr])\n\t"
            "vle32.v v12, (%[aptr2])\n\t"
            "add %[bptr2], %[bptr2], %[cs_c]\n\t"
            "add %[cptr], %[cptr], %[cs_c]\n\t"
            "vfmacc.vf v17,  f1, v9\n\t"
            "vfmacc.vf v18,  f1, v10\n\t"

            "add %[aptr], %[aptr], %[cs_c]\n\t"
            "add %[aptr2], %[aptr2], %[cs_c]\n\t"
            "vse32.v v17, (%[bptr2])\n\t"
            "vse32.v v18, (%[cptr])\n\t"
            "vle32.v v13, (%[aptr])\n\t"
            "vle32.v v14, (%[aptr2])\n\t"
            "add %[bptr2], %[bptr2], %[cs_c]\n\t"
            "add %[cptr], %[cptr], %[cs_c]\n\t"
            "vfmacc.vf v19,  f1, v11\n\t"
            "vfmacc.vf v20,  f1, v12\n\t"

            "add %[aptr], %[aptr], %[cs_c]\n\t"
            "add %[aptr2], %[aptr2], %[cs_c]\n\t"
            "vse32.v v19, (%[bptr2])\n\t"
            "vse32.v v20, (%[cptr])\n\t"
            "vle32.v v15, (%[aptr])\n\t"
            "vle32.v v16, (%[aptr2])\n\t"
            "add %[bptr2], %[bptr2], %[cs_c]\n\t"
            "add %[cptr], %[cptr], %[cs_c]\n\t"
            "vfmacc.vf v21,  f1, v13\n\t"
            "vfmacc.vf v22,  f1, v14\n\t"

            "add %[aptr], %[aptr], %[cs_c]\n\t"
            "add %[aptr2], %[aptr2], %[cs_c]\n\t"
            "vse32.v v21, (%[bptr2])\n\t"
            "vse32.v v22, (%[cptr])\n\t"
            "vle32.v v17, (%[aptr])\n\t"
            "vle32.v v18, (%[aptr2])\n\t"
            "add %[bptr2], %[bptr2], %[cs_c]\n\t"
            "add %[cptr], %[cptr], %[cs_c]\n\t"
            "vfmacc.vf v23,  f1, v15\n\t"
            "vfmacc.vf v24,  f1, v16\n\t"

            "add %[aptr], %[aptr], %[cs_c]\n\t"
            "add %[aptr2], %[aptr2], %[cs_c]\n\t"
            "vse32.v v23, (%[bptr2])\n\t"
            "vse32.v v24, (%[cptr])\n\t"
            "vle32.v v19, (%[aptr])\n\t"
            "vle32.v v20, (%[aptr2])\n\t"
            "add %[bptr2], %[bptr2], %[cs_c]\n\t"
            "add %[cptr], %[cptr], %[cs_c]\n\t"
            "vfmacc.vf v25,  f1, v17\n\t"
            "vfmacc.vf v26,  f1, v18\n\t"

            "vse32.v v25, (%[bptr2])\n\t"
            "vse32.v v26, (%[cptr])\n\t"
            "add %[bptr2], %[bptr2], %[cs_c]\n\t"
            "add %[cptr], %[cptr], %[cs_c]\n\t"
            "vfmacc.vf v27,  f1, v19\n\t"
            "vfmacc.vf v28,  f1, v20\n\t"

            "vse32.v v27, (%[bptr2])\n\t"
            "vse32.v v28, (%[cptr])\n\t"


            "j .sgemm_ukr_end%=\n\t"
        ".betazero%=:\n\t" // TODO: written manually and unoptimized
            
            "vse32.v v1, (%[bptr2])\n\t"
            "vse32.v v2, (%[cptr])\n\t"
            "add %[bptr2], %[bptr2], %[cs_c]\n\t"
            "add %[cptr], %[cptr], %[cs_c]\n\t"

            "vse32.v v3, (%[bptr2])\n\t"
            "vse32.v v4, (%[cptr])\n\t"
            "add %[bptr2], %[bptr2], %[cs_c]\n\t"
            "add %[cptr], %[cptr], %[cs_c]\n\t"

            "vse32.v v5, (%[bptr2])\n\t"
            "vse32.v v6, (%[cptr])\n\t"
            "add %[bptr2], %[bptr2], %[cs_c]\n\t"
            "add %[cptr], %[cptr], %[cs_c]\n\t"

            "vse32.v v7, (%[bptr2])\n\t"
            "vse32.v v8, (%[cptr])\n\t"
            "add %[bptr2], %[bptr2], %[cs_c]\n\t"
            "add %[cptr], %[cptr], %[cs_c]\n\t"

            "vse32.v v9, (%[bptr2])\n\t"
            "vse32.v v10, (%[cptr])\n\t"
            "add %[bptr2], %[bptr2], %[cs_c]\n\t"
            "add %[cptr], %[cptr], %[cs_c]\n\t"

            "vse32.v v11, (%[bptr2])\n\t"
            "vse32.v v12, (%[cptr])\n\t"
            "add %[bptr2], %[bptr2], %[cs_c]\n\t"
            "add %[cptr], %[cptr], %[cs_c]\n\t"

            "vse32.v v13, (%[bptr2])\n\t"
            "vse32.v v14, (%[cptr])\n\t"
            "add %[bptr2], %[bptr2], %[cs_c]\n\t"
            "add %[cptr], %[cptr], %[cs_c]\n\t"

            "vse32.v v15, (%[bptr2])\n\t"
            "vse32.v v16, (%[cptr])\n\t"
            "add %[bptr2], %[bptr2], %[cs_c]\n\t"
            "add %[cptr], %[cptr], %[cs_c]\n\t"

            "vse32.v v17, (%[bptr2])\n\t"
            "vse32.v v18, (%[cptr])\n\t"
            "add %[bptr2], %[bptr2], %[cs_c]\n\t"
            "add %[cptr], %[cptr], %[cs_c]\n\t"

            "vse32.v v19, (%[bptr2])\n\t"
            "vse32.v v20, (%[cptr])\n\t"
            "add %[bptr2], %[bptr2], %[cs_c]\n\t"
            "add %[cptr], %[cptr], %[cs_c]\n\t"

            "vse32.v v21, (%[bptr2])\n\t"
            "vse32.v v22, (%[cptr])\n\t"
            "add %[bptr2], %[bptr2], %[cs_c]\n\t"
            "add %[cptr], %[cptr], %[cs_c]\n\t"

            "vse32.v v23, (%[bptr2])\n\t"
            "vse32.v v24, (%[cptr])\n\t"
            "add %[bptr2], %[bptr2], %[cs_c]\n\t"
            "add %[cptr], %[cptr], %[cs_c]\n\t"

            "vse32.v v25, (%[bptr2])\n\t"
            "vse32.v v26, (%[cptr])\n\t"
            "add %[bptr2], %[bptr2], %[cs_c]\n\t"
            "add %[cptr], %[cptr], %[cs_c]\n\t"

            "vse32.v v27, (%[bptr2])\n\t"
            "vse32.v v28, (%[cptr])\n\t"
            
        ".sgemm_ukr_end%=:\n\t"

        "prefetch.r 0(%[a_next])\n\t"
        "prefetch.r 0(%[b_next])\n\t"

        : [dummy_c] "+m"(*(float(*)[])c),
          [unroll] "=r" (unroll), [vlenx2] "=r" (vlenx2),
          [aptr] "+r" (a), [aptr2] "=r" (aptr2),
          [bptr] "+r" (b), [bptr2] "=r" (bptr2),
          [cptr] "+r" (c), [cptr2] "=r" (cptr2),
          [vlen] "+r" (vlen), [n] "+r" (k), [counter] "=r" (counter),
          [rs_c] "+r" (rs_c), [cs_c] "+r" (cs_c)
        : [alpha] "r" (alpha), [beta] "r" (beta), [pfdist] "r" (pfdist),
          [a_next] "r" (a_next), [b_next] "r" (b_next)
        : "f0","f1",
          "f2","f3","f4","f5","f6","f7","f8","f9",
          "f10","f11","f12","f13","f14","f15","f16","f17",
          "f18","f19","f20","f21","f22","f23","f24","f25",
          "f26","f27","f28","f29",
          "v1","v2","v3","v4","v5","v6","v7","v8",
          "v9","v10","v11","v12","v13","v14","v15","v16",
          "v17","v18","v19","v20","v21","v22","v23","v24",
          "v25","v26","v27","v28","v29","v30","v31","v0"
    );


    GEMM_UKR_FLUSH_CT( s );
}
