/* file: service_blas_mkl.h */
/*******************************************************************************
* Copyright 2014 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/*
//++
//  Template wrappers for common Intel(R) MKL functions.
//--
*/

#ifndef __SERVICE_BLAS_MKL_H__
#define __SERVICE_BLAS_MKL_H__

#include "services/daal_defines.h"
#include <mkl.h>
#include "mkl_daal.h"

namespace daal
{
namespace internal
{
namespace mkl
{
template <typename fpType, CpuType cpu>
struct MklBlas
{};

/*
// Double precision functions definition
*/

template <CpuType cpu>
struct MklBlas<double, cpu>
{
    typedef DAAL_INT SizeType;

    static void xsyrk(char * uplo, char * trans, DAAL_INT * p, DAAL_INT * n, double * alpha, double * a, DAAL_INT * lda, double * beta, double * ata,
                      DAAL_INT * ldata)
    {
        __DAAL_MKLFN_CALL(blas_, dsyrk, (uplo, trans, (MKL_INT *)p, (MKL_INT *)n, alpha, a, (MKL_INT *)lda, beta, ata, (MKL_INT *)ldata));
    }

    static void xxsyrk(char * uplo, char * trans, DAAL_INT * p, DAAL_INT * n, double * alpha, double * a, DAAL_INT * lda, double * beta, double * ata,
                       DAAL_INT * ldata)
    {
        __DAAL_MKLFN_CALL(blas_, dsyrk, (uplo, trans, (MKL_INT *)p, (MKL_INT *)n, alpha, a, (MKL_INT *)lda, beta, ata, (MKL_INT *)ldata));
    }

    static void xsyr(const char * uplo, const DAAL_INT * n, const double * alpha, const double * x, const DAAL_INT * incx, double * a,
                     const DAAL_INT * lda)
    {
        __DAAL_MKLFN_CALL(blas_, dsyr, (uplo, (MKL_INT *)n, alpha, x, (MKL_INT *)incx, a, (MKL_INT *)lda));
    }

    static void xxsyr(const char * uplo, const DAAL_INT * n, const double * alpha, const double * x, const DAAL_INT * incx, double * a,
                      const DAAL_INT * lda)
    {
        int old_threads = mkl_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(blas_, dsyr, (uplo, (MKL_INT *)n, alpha, x, (MKL_INT *)incx, a, (MKL_INT *)lda));
        mkl_serv_set_num_threads_local(old_threads);
    }

    static void xgemm(const char * transa, const char * transb, const DAAL_INT * p, const DAAL_INT * ny, const DAAL_INT * n, const double * alpha,
                      const double * a, const DAAL_INT * lda, const double * y, const DAAL_INT * ldy, const double * beta, double * aty,
                      const DAAL_INT * ldaty)
    {
        __DAAL_MKLFN_CALL(
            blas_, xdgemm,
            (transa, transb, (MKL_INT *)p, (MKL_INT *)ny, (MKL_INT *)n, alpha, a, (MKL_INT *)lda, y, (MKL_INT *)ldy, beta, aty, (MKL_INT *)ldaty));
    }

    static void xxgemm(const char * transa, const char * transb, const DAAL_INT * p, const DAAL_INT * ny, const DAAL_INT * n, const double * alpha,
                       const double * a, const DAAL_INT * lda, const double * y, const DAAL_INT * ldy, const double * beta, double * aty,
                       const DAAL_INT * ldaty)
    {
        __DAAL_MKLFN_CALL(
            blas_, xdgemm,
            (transa, transb, (MKL_INT *)p, (MKL_INT *)ny, (MKL_INT *)n, alpha, a, (MKL_INT *)lda, y, (MKL_INT *)ldy, beta, aty, (MKL_INT *)ldaty));
    }

    static void xsymm(const char * side, const char * uplo, const DAAL_INT * m, const DAAL_INT * n, const double * alpha, const double * a,
                      const DAAL_INT * lda, const double * b, const DAAL_INT * ldb, const double * beta, double * c, const DAAL_INT * ldc)
    {
        __DAAL_MKLFN_CALL(blas_, dsymm,
                          (side, uplo, (MKL_INT *)m, (MKL_INT *)n, alpha, a, (MKL_INT *)lda, b, (MKL_INT *)ldb, beta, c, (MKL_INT *)ldc));
    }

    static void xxsymm(char * side, char * uplo, DAAL_INT * m, DAAL_INT * n, double * alpha, double * a, DAAL_INT * lda, double * b, DAAL_INT * ldb,
                       double * beta, double * c, DAAL_INT * ldc)
    {
        int old_threads = mkl_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(blas_, dsymm,
                          (side, uplo, (MKL_INT *)m, (MKL_INT *)n, alpha, a, (MKL_INT *)lda, b, (MKL_INT *)ldb, beta, c, (MKL_INT *)ldc));
        mkl_serv_set_num_threads_local(old_threads);
    }

    static void xgemv(const char * trans, const DAAL_INT * m, const DAAL_INT * n, const double * alpha, const double * a, const DAAL_INT * lda,
                      const double * x, const DAAL_INT * incx, const double * beta, double * y, const DAAL_INT * incy)
    {
        __DAAL_MKLFN_CALL(blas_, dgemv, (trans, (MKL_INT *)m, (MKL_INT *)n, alpha, a, (MKL_INT *)lda, x, (MKL_INT *)incx, beta, y, (MKL_INT *)incy));
    }

    static void xxgemv(const char * trans, const DAAL_INT * m, const DAAL_INT * n, const double * alpha, const double * a, const DAAL_INT * lda,
                       const double * x, const DAAL_INT * incx, const double * beta, double * y, const DAAL_INT * incy)
    {
        int old_threads = mkl_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(blas_, dgemv, (trans, (MKL_INT *)m, (MKL_INT *)n, alpha, a, (MKL_INT *)lda, x, (MKL_INT *)incx, beta, y, (MKL_INT *)incy));
        mkl_serv_set_num_threads_local(old_threads);
    }

    static void xaxpy(DAAL_INT * n, double * a, double * x, DAAL_INT * incx, double * y, DAAL_INT * incy)
    {
        __DAAL_MKLFN_CALL(blas_, daxpy, ((MKL_INT *)n, a, x, (MKL_INT *)incx, y, (MKL_INT *)incy));
    }

    static void xxaxpy(const DAAL_INT * n, const double * a, const double * x, const DAAL_INT * incx, double * y, const DAAL_INT * incy)
    {
        int old_threads = mkl_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(blas_, daxpy, ((MKL_INT *)n, a, x, (MKL_INT *)incx, y, (MKL_INT *)incy));
        mkl_serv_set_num_threads_local(old_threads);
    }

    static double xxdot(const DAAL_INT * n, const double * x, const DAAL_INT * incx, const double * y, const DAAL_INT * incy)
    {
        __DAAL_MKLFN_CALL_RETURN(blas_, ddot, ((MKL_INT *)n, x, (MKL_INT *)incx, y, (MKL_INT *)incy));
        return 0;
    }
};

/*
// Single precision functions definition
*/

template <CpuType cpu>
struct MklBlas<float, cpu>
{
    typedef DAAL_INT SizeType;

    static void xsyrk(char * uplo, char * trans, DAAL_INT * p, DAAL_INT * n, float * alpha, float * a, DAAL_INT * lda, float * beta, float * ata,
                      DAAL_INT * ldata)
    {
        __DAAL_MKLFN_CALL(blas_, ssyrk, (uplo, trans, (MKL_INT *)p, (MKL_INT *)n, alpha, a, (MKL_INT *)lda, beta, ata, (MKL_INT *)ldata));
    }

    static void xxsyrk(char * uplo, char * trans, DAAL_INT * p, DAAL_INT * n, float * alpha, float * a, DAAL_INT * lda, float * beta, float * ata,
                       DAAL_INT * ldata)
    {
        int old_threads = mkl_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(blas_, ssyrk, (uplo, trans, (MKL_INT *)p, (MKL_INT *)n, alpha, a, (MKL_INT *)lda, beta, ata, (MKL_INT *)ldata));
        mkl_serv_set_num_threads_local(old_threads);
    }

    static void xsyr(const char * uplo, const DAAL_INT * n, const float * alpha, const float * x, const DAAL_INT * incx, float * a,
                     const DAAL_INT * lda)
    {
        __DAAL_MKLFN_CALL(blas_, ssyr, (uplo, (MKL_INT *)n, alpha, x, (MKL_INT *)incx, a, (MKL_INT *)lda));
    }

    static void xxsyr(const char * uplo, const DAAL_INT * n, const float * alpha, const float * x, const DAAL_INT * incx, float * a,
                      const DAAL_INT * lda)
    {
        int old_threads = mkl_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(blas_, ssyr, (uplo, (MKL_INT *)n, alpha, x, (MKL_INT *)incx, a, (MKL_INT *)lda));
        mkl_serv_set_num_threads_local(old_threads);
    }

    static void xgemm(const char * transa, const char * transb, const DAAL_INT * p, const DAAL_INT * ny, const DAAL_INT * n, const float * alpha,
                      const float * a, const DAAL_INT * lda, const float * y, const DAAL_INT * ldy, const float * beta, float * aty,
                      const DAAL_INT * ldaty)
    {
        __DAAL_MKLFN_CALL_(
            blas_, sgemm,
            (transa, transb, (MKL_INT *)p, (MKL_INT *)ny, (MKL_INT *)n, alpha, a, (MKL_INT *)lda, y, (MKL_INT *)ldy, beta, aty, (MKL_INT *)ldaty));
    }

    static void xxgemm(const char * transa, const char * transb, const DAAL_INT * p, const DAAL_INT * ny, const DAAL_INT * n, const float * alpha,
                       const float * a, const DAAL_INT * lda, const float * y, const DAAL_INT * ldy, const float * beta, float * aty,
                       const DAAL_INT * ldaty)
    {
        __DAAL_MKLFN_CALL_(
            blas_, sgemm,
            (transa, transb, (MKL_INT *)p, (MKL_INT *)ny, (MKL_INT *)n, alpha, a, (MKL_INT *)lda, y, (MKL_INT *)ldy, beta, aty, (MKL_INT *)ldaty));
    }

    static void xsymm(const char * side, const char * uplo, const DAAL_INT * m, const DAAL_INT * n, const float * alpha, const float * a,
                      const DAAL_INT * lda, const float * b, const DAAL_INT * ldb, const float * beta, float * c, const DAAL_INT * ldc)
    {
        __DAAL_MKLFN_CALL(blas_, ssymm,
                          (side, uplo, (MKL_INT *)m, (MKL_INT *)n, alpha, a, (MKL_INT *)lda, b, (MKL_INT *)ldb, beta, c, (MKL_INT *)ldc));
    }

    static void xxsymm(char * side, char * uplo, DAAL_INT * m, DAAL_INT * n, float * alpha, float * a, DAAL_INT * lda, float * b, DAAL_INT * ldb,
                       float * beta, float * c, DAAL_INT * ldc)
    {
        int old_threads = mkl_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(blas_, ssymm,
                          (side, uplo, (MKL_INT *)m, (MKL_INT *)n, alpha, a, (MKL_INT *)lda, b, (MKL_INT *)ldb, beta, c, (MKL_INT *)ldc));
        mkl_serv_set_num_threads_local(old_threads);
    }

    static void xgemv(const char * trans, const DAAL_INT * m, const DAAL_INT * n, const float * alpha, const float * a, const DAAL_INT * lda,
                      const float * x, const DAAL_INT * incx, const float * beta, float * y, const DAAL_INT * incy)
    {
        __DAAL_MKLFN_CALL(blas_, sgemv, (trans, (MKL_INT *)m, (MKL_INT *)n, alpha, a, (MKL_INT *)lda, x, (MKL_INT *)incx, beta, y, (MKL_INT *)incy));
    }

    static void xxgemv(const char * trans, const DAAL_INT * m, const DAAL_INT * n, const float * alpha, const float * a, const DAAL_INT * lda,
                       const float * x, const DAAL_INT * incx, const float * beta, float * y, const DAAL_INT * incy)
    {
        int old_threads = mkl_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(blas_, sgemv, (trans, (MKL_INT *)m, (MKL_INT *)n, alpha, a, (MKL_INT *)lda, x, (MKL_INT *)incx, beta, y, (MKL_INT *)incy));
        mkl_serv_set_num_threads_local(old_threads);
    }

    static void xaxpy(DAAL_INT * n, float * a, float * x, DAAL_INT * incx, float * y, DAAL_INT * incy)
    {
        __DAAL_MKLFN_CALL(blas_, saxpy, ((MKL_INT *)n, a, x, (MKL_INT *)incx, y, (MKL_INT *)incy));
    }

    static void xxaxpy(const DAAL_INT * n, const float * a, const float * x, const DAAL_INT * incx, float * y, const DAAL_INT * incy)
    {
        int old_threads = mkl_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(blas_, saxpy, ((MKL_INT *)n, a, x, (MKL_INT *)incx, y, (MKL_INT *)incy));
        mkl_serv_set_num_threads_local(old_threads);
    }

    static float xxdot(const DAAL_INT * n, const float * x, const DAAL_INT * incx, const float * y, const DAAL_INT * incy)
    {
        __DAAL_MKLFN_CALL_RETURN(blas_, sdot, ((MKL_INT *)n, x, (MKL_INT *)incx, y, (MKL_INT *)incy));
        return 0;
    }
};

} // namespace mkl
} // namespace internal
} // namespace daal

#endif
