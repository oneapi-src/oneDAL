/* file: env_detect.cpp */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/*
//++
//  Definitions of structures used for environment detection.
//--
*/

#include <immintrin.h>

#include "env_detect.h"
#include "daal_defines.h"
#include "service_defines.h"
#include "service_service.h"
#include "threading.h"
#include "error_indexes.h"

#include "service_topo.h"
#include "service_thread_pinner.h"

static daal::services::Environment::LibraryThreadingType daal_thr_set = (daal::services::Environment::LibraryThreadingType) - 1;
static bool isInit = false;

namespace daal
{
namespace services
{
void daal_free_buffers();
}
}

DAAL_EXPORT daal::services::Environment *daal::services::Environment::getInstance()
{
    static daal::services::Environment instance;
    return &instance;
}

DAAL_EXPORT int daal::services::Environment::freeInstance()
{
    return 0;
}

DAAL_EXPORT int daal::services::Environment::getCpuId(int enable)
{
    initNumberOfThreads();
    if(!_env.cpuid_init_flag)
    {
        _cpu_detect(enable);
        _env.cpuid_init_flag = true;
    }

    return static_cast<int>(_env.cpuid);
}

DAAL_EXPORT int daal::services::Environment::enableInstructionsSet(int enable)
{
    initNumberOfThreads();
    if(!_env.cpuid_init_flag)
    {
        _cpu_detect(enable);
        _env.cpuid_init_flag = true;
    }

    return static_cast<int>(_env.cpuid);
}

DAAL_EXPORT int daal::services::Environment::setCpuId(int cpuid)
{
    initNumberOfThreads();
    int host_cpuid=__daal_serv_cpu_detect(daal::services::Environment::avx512_mic_e1);

    if(!_env.cpuid_init_flag)
    {
        if (-1 == _env.cpuid) {
            if(cpuid > daal::lastCpuType || cpuid < 0)
                return daal::services::ErrorCpuIsInvalid;

            if(cpuid > host_cpuid)
            {
                _cpu_detect(daal::services::Environment::avx512_mic_e1);
            }
            else
            {
                if(daal::avx512_mic == cpuid && daal::avx512_mic != host_cpuid)
                {
                    _cpu_detect(daal::services::Environment::avx512_mic_e1);
                }
                else
                {
                    _env.cpuid = cpuid;
                }
            }
        }

        _env.cpuid_init_flag = true;
    }

    return static_cast<int>(_env.cpuid);
}

daal::services::Environment::LibraryThreadingType __daal_serv_get_thr_set()
{
    return daal_thr_set;
}

DAAL_EXPORT void daal::services::Environment::setDynamicLibraryThreadingTypeOnWindows(daal::services::Environment::LibraryThreadingType thr)
{
    daal_thr_set = thr;
    initNumberOfThreads();
}

DAAL_EXPORT daal::services::Environment::Environment() : _init(0)
{
    _env.cpuid_init_flag = false;
    _env.cpuid = -1;
}

DAAL_EXPORT daal::services::Environment::Environment(const Environment& e) : _init(0)
{
    _env.cpuid_init_flag = false;
    _env.cpuid = -1;
}

DAAL_EXPORT void daal::services::Environment::initNumberOfThreads()
{
    if ( isInit ) return;

    /* if HT enabled - set _numThreads to physical cores num */
    if( daal::internal::Service<>::serv_get_ht() )
    {
        /* Number of cores = number of cpu packages * number of cores per cpu package */
        int ncores = daal::internal::Service<>::serv_get_ncpus() * daal::internal::Service<>::serv_get_ncorespercpu();

        /*  Re-set number of threads if ncores is valid and different to _numThreads */
        if( (ncores > 0) && (ncores < _daal_threader_get_max_threads()) )
        {
            daal::services::Environment::setNumberOfThreads(ncores);
        }
    }
    isInit = true;
}

DAAL_EXPORT daal::services::Environment::~Environment()
{
    daal::services::daal_free_buffers();
    _daal_tbb_task_scheduler_free(_init);
}

void daal::services::Environment::_cpu_detect(int enable)
{
    initNumberOfThreads();
    if (-1 == _env.cpuid) {
        _env.cpuid = __daal_serv_cpu_detect(enable);
    }
}

DAAL_EXPORT void daal::services::Environment::setNumberOfThreads(const size_t numThreads)
{
    isInit = true;
    daal::setNumberOfThreads(numThreads, &_init);
}

DAAL_EXPORT size_t daal::services::Environment::getNumberOfThreads() const { return daal::threader_get_threads_number(); }

DAAL_EXPORT int daal::services::Environment::setMemoryLimit(MemType type, size_t limit) {
    initNumberOfThreads();
    return daal::internal::Service<>::serv_set_memory_limit(type, limit);
}


DAAL_EXPORT void daal::services::Environment::enableThreadPinning(const bool enableThreadPinningFlag)
{
    initNumberOfThreads();
#if !(defined DAAL_THREAD_PINNING_DISABLED)
    daal::services::internal::thread_pinner_t*  thread_pinner = daal::services::internal::getThreadPinner(true, read_topology, delete_topology);

    if(thread_pinner != NULL)
    {
        thread_pinner->set_pinning(enableThreadPinningFlag);
    }
#endif
    return;
}
