/* file: engine_types_internal.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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

//++
//  Implementation of initializer types.
//--

#ifndef __ENGINE_TYPES_INTERNAL_H__
#define __ENGINE_TYPES_INTERNAL_H__

#include "engines/engine_batch_impl.h"
#include "algorithms/engines/engine_family.h"
#include "service_arrays.h"

namespace daal
{
namespace algorithms
{
namespace engines
{

namespace internal
{

template<CpuType cpu>
struct Params
{
    size_t numberOfStreams;
    services::internal::TArray<size_t, cpu> nSkip;

    Params(size_t numStreams) : numberOfStreams(numStreams), nSkip(numStreams) {}
};

template<CpuType cpu>
class EnginesCollection
{
private:
    engines::FamilyEnginePtr _clonedEngine;
    ParallelizationTechnique _technique;
    size_t _numberOfStreams;

public:
    EnginesCollection(engines::EnginePtr engine,
                      ParallelizationTechnique technique,
                      Params<cpu> &params,
                      services::internal::TArray<engines::EnginePtr, cpu> &engines, services::Status *st)
    {
        *st = initEngines(engine, technique, params, engines);
    }

    engines::EnginePtr getUpdatedEngine(engines::EnginePtr engine,
                                        services::internal::TArray<engines::EnginePtr, cpu> &engines,
                                        services::internal::TArray<size_t, cpu> &numElems)
    {
        switch(_technique)
        {
            case skipahead:
            {
                return engines[_numberOfStreams - 1]->clone();
            }
            case leapfrog:
            {
                size_t nSkip = 0;
                for(size_t i = 0; i < _numberOfStreams; i++)
                {
                    if(nSkip < numElems[i])
                    {
                        nSkip = numElems[i];
                    }
                }
                auto updatedEngine = engine->clone();
                updatedEngine->skipAhead(nSkip);
                return updatedEngine;
            }
            case family: return _clonedEngine;
        }
        return engines::EnginePtr();
    }

    services::Status initEngines(engines::EnginePtr engine,
                                 ParallelizationTechnique technique,
                                 Params<cpu> &params,
                                 services::internal::TArray<engines::EnginePtr, cpu> &engines)
    {
        _technique = technique;
        _numberOfStreams = params.numberOfStreams;

        auto engineImpl = dynamic_cast<engines::internal::BatchBaseImpl*>(engine.get());
        DAAL_CHECK(engineImpl, ErrorEngineNotSupported);
        DAAL_CHECK(engineImpl->hasSupport(_technique), ErrorEngineNotSupported);
        switch(_technique)
        {
            case skipahead:
            {
                for(size_t i = 0; i < _numberOfStreams; i++)
                {
                    auto engineLocal = engine->clone();
                    DAAL_CHECK_STATUS_VAR(engineLocal->skipAhead(params.nSkip[i]));
                    engines[i] = engineLocal;
                }
                break;
            }
            case leapfrog:
            {
                for(size_t i = 0; i < _numberOfStreams; i++)
                {
                    auto engineLocal = engine->clone();
                    DAAL_CHECK_STATUS_VAR(engineLocal->leapfrog(i, params.numberOfStreams));
                    engines[i] = engineLocal;
                }
                break;
            }
            case family:
            {
                _clonedEngine = services::dynamicPointerCast<engines::FamilyBatchBase>(engine->clone());
                DAAL_CHECK(_clonedEngine, ErrorEngineNotSupported);
                DAAL_CHECK(_clonedEngine->getMaxNumberOfStreams() >= _numberOfStreams, ErrorEngineNotSupported);
                size_t numStreams = _clonedEngine->getNumberOfStreams();
                if(numStreams < _numberOfStreams)
                {
                    DAAL_CHECK_STATUS_VAR(_clonedEngine->add(_numberOfStreams - numStreams)); // silently initialize more independent streams
                }
                for(size_t i = 0; i < _numberOfStreams; i++)
                {
                    engines[i] = _clonedEngine->get(i);
                }
            }
        }
        return services::Status();
    }
};

} // namespace internal
} // namespace engines
} // namespace algorithms
} // namespace daal

#endif // __ENGINE_TYPES_INTERNAL_H__
