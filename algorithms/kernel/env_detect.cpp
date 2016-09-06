/* file: env_detect.cpp */
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
//  Definitions of structures used for environment detection.
//--
*/

#include <immintrin.h>

#include "env_detect.h"
#include "daal_defines.h"
#include "service_defines.h"
#include "mkl_daal.h"
#include "threading.h"

#if defined(_MSC_VER) && !defined(__DAAL_IMPLEMENTATION)
    #pragma comment(lib, "tbb")
#endif

static daal::services::Environment::LibraryThreadingType daal_thr_set = (daal::services::Environment::LibraryThreadingType) - 1;

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
    if(!_env.cpuid_init_flag)
    {
        _cpu_detect(enable);
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
}

DAAL_EXPORT daal::services::Environment::Environment() : _init(0)
{
    _env.cpuid_init_flag = false;
    _env.cpuid = -1;

    _numThreads = fpk_serv_get_max_threads();

    /* if HT enabled - set _numThreads to physical cores num */
    if( fpk_serv_get_ht() )
    {
        /* Number of cores = number of cpu packages * number of cores per cpu package */
        int ncores = fpk_serv_get_ncpus() * fpk_serv_get_ncorespercpu();

        /*  Re-set number of threads if ncores is valid and different to _numThreads */
        if( (ncores > 0) && (ncores < _numThreads) )
        {
            _numThreads = ncores;
            fpk_serv_set_num_threads(_numThreads);
        }
    }
}

DAAL_EXPORT daal::services::Environment::Environment(const Environment& e) : _init(0)
{
    _env.cpuid_init_flag = false;
    _env.cpuid = -1;

    _numThreads = e.getNumberOfThreads();
}

DAAL_EXPORT daal::services::Environment::~Environment() {}

void daal::services::Environment::_cpu_detect(int enable)
{
    if (-1 == _env.cpuid) {
        _env.cpuid = __daal_serv_cpu_detect(enable);
    }
}

DAAL_EXPORT void daal::services::Environment::setNumberOfThreads(const size_t numThreads)
{
    _numThreads = daal::setNumberOfThreads(numThreads, &_init);
    fpk_serv_set_num_threads(_numThreads);
}

DAAL_EXPORT size_t daal::services::Environment::getNumberOfThreads() const { return _numThreads; }

DAAL_EXPORT int daal::services::Environment::setMemoryLimit(MemType type, size_t limit) {
    return fpk_serv_set_memory_limit(type, limit);
}
