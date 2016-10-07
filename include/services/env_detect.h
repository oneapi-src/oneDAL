/* file: env_detect.h */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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
//  Implementation of the class used for environment detection.
//--
*/

#ifndef __ENV_DETECT_H__
#define __ENV_DETECT_H__

#include "services/base.h"
#include "services/daal_defines.h"

namespace daal
{
/**
 * @defgroup env_detect Managing the Computational Environment
 * \brief Provides methods to interact with the environment, including processor detection and control by the number of threads.
 * @ingroup services
 * @{
 */
/**
 * <a name="DAAL-ENUM-CPUTYPE"></a>
 * Supported types of processor architectures
 *
 * \ref opt_notice
 */
enum CpuType
{
    sse2        = 0,    /*!< Intel(R) Streaming SIMD Extensions 2 (Intel(R) SSE2) */
    ssse3       = 1,    /*!< Intel(R) Supplemental Streaming SIMD Extensions 3 (Intel(R) SSSE3) */
    sse42       = 2,    /*!< Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) */
    avx         = 3,    /*!< Intel(R) Advanced Vector Extensions (Intel(R) AVX) */
    avx2        = 4,    /*!< Intel(R) Advanced Vector Extensions 2 (Intel(R) AVX2) */
    avx512_mic  = 5,    /*!< Intel(R) Xeon Phi(TM) processors/coprocessors based on Intel(R) Advanced Vector Extensions 512
                             (Intel(R) AVX512) */
    avx512      = 6     /*!< Intel(R) Xeon(R) processors based on Intel AVX-512 */
};

namespace services
{
namespace interface1
{

/**
 * <a name="DAAL-CLASS-SERVICES__ENVIRONMENT"></a>
 * \brief Class that provides methods to interact with the environment, including processor detection and control by the number of threads
 *
 * \ref opt_notice
 */
class DAAL_EXPORT Environment : public Base
{
public:
    /**
    * <a name="DAAL-CLASS-_ENVSTRUCT"></a>
    * \brief The environment structure
    */
    typedef struct _envStruct
    {
        bool cpuid_init_flag;
        size_t cpuid;
    } env;

    /**
     *  Returns the environment instance
     *  \return The environment instance
     */
    static Environment *getInstance();

    /**
     *  Decreases the instance counter
     *  \return The return code
     */
    static int freeInstance();

    /**
     * <a name="DAAL-ENUM-SERVICES__CPUTYPEENABLE"></a>
     * \brief CPU types
     */
    enum CpuTypeEnable
    {
        cpu_default = 0,    /*!< Default processor type */
        avx512_mic = 1,     /*!< Intel(R) Xeon Phi(TM) processors/coprocessors based on Intel(R) Advanced Vector Extensions 512 (Intel(R) AVX-512) */
        avx512 = 2          /*!< Intel(R) Xeon(R) processors based on Intel AVX-512 */
    };

    /**
    *  Detects the processor type
    *  \param[in] enable  An enabling flag
    *  \return The CPU ID
    */
    int getCpuId(int enable = cpu_default);

    /**
     * <a name="DAAL-ENUM-SERVICES__LIBRARYTHREADINGTYPE"></a>
     * The threading mode of the library
     */
    enum LibraryThreadingType
    {
        MultiThreaded = 0,  /*!< Multi-threaded mode */
        SingleThreaded = 1  /*!< Single-threaded mode */
    };

    /**
    *  Sets the threading mode on Windows*
    *  \param[in] type  The threading mode of the library
    */
    void setDynamicLibraryThreadingTypeOnWindows( LibraryThreadingType type );

    /**
    *  Sets the number of threads to use
    *  \param[in] numThreads   The number of threads
    */
    void setNumberOfThreads(const size_t numThreads);

    /**
    *  Returns the number of used threads
    *  \return The number of used threads
    */
    size_t getNumberOfThreads() const;

    /**
     * Limits the amount of memory of the given type available to internal function calls
     * \param[in] type   Memory type
     * \param[in] limit  Limit in megabytes
     */
    int setMemoryLimit(MemType type, size_t limit);

private:
    Environment();
    Environment(const Environment &e);
    ~Environment();

    void _cpu_detect(int);

    env _env;

    size_t _numThreads;
    void *_init;
};
} // namespace interface1

using interface1::Environment;

}
/** @} */
}
#endif
