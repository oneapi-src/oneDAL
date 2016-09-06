/* file: init_types.i */
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

#include "daal.h"

#include "common_defines.i"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::algorithms::kmeans::init;
using namespace daal::services;

#define DISPATCHED_CALL(result,addr,call,op,cast,precision,method,cmode,step)                                   \
    if(cmode == jBatch)                                                                                         \
    {                                                                                                           \
        if(precision == 0)                                                                                      \
        {                                                                                                       \
            if(method == deterministicDense)                                                                    \
            {                                                                                                   \
                SharedPtr<Batch<double, deterministicDense> > alg =                                             \
                    staticPointerCast<Batch<double, deterministicDense>, AlgorithmIface>                        \
                        (*(SharedPtr<AlgorithmIface> *)addr);                                                   \
                result = cast(op(alg->call));                                                                   \
            } else if(method == randomDense)                                                                    \
            {                                                                                                   \
                SharedPtr<Batch<double, randomDense> > alg =                                                    \
                    staticPointerCast<Batch<double, randomDense>, AlgorithmIface>                               \
                        (*(SharedPtr<AlgorithmIface> *)addr);                                                   \
                result = cast(op(alg->call));                                                                   \
            } else if(method == deterministicCSR)                                                               \
            {                                                                                                   \
                SharedPtr<Batch<double, deterministicCSR> > alg =                                               \
                    staticPointerCast<Batch<double, deterministicCSR>, AlgorithmIface>                          \
                        (*(SharedPtr<AlgorithmIface> *)addr);                                                   \
                result = cast(op(alg->call));                                                                   \
            } else if(method == randomCSR)                                                                      \
            {                                                                                                   \
                SharedPtr<Batch<double, randomCSR> > alg =                                                      \
                    staticPointerCast<Batch<double, randomCSR>, AlgorithmIface>                                 \
                        (*(SharedPtr<AlgorithmIface> *)addr);                                                   \
                result = cast(op(alg->call));                                                                   \
            }                                                                                                   \
        }                                                                                                       \
        else                                                                                                    \
        {                                                                                                       \
            if(method == deterministicDense)                                                                    \
            {                                                                                                   \
                SharedPtr<Batch<float, deterministicDense> > alg =                                              \
                    staticPointerCast<Batch<float, deterministicDense>, AlgorithmIface>                         \
                        (*(SharedPtr<AlgorithmIface> *)addr);                                                   \
                result = cast(op(alg->call));                                                                   \
            } else if(method == randomDense)                                                                    \
            {                                                                                                   \
                SharedPtr<Batch<float, randomDense> > alg =                                                     \
                    staticPointerCast<Batch<float, randomDense>, AlgorithmIface>                                \
                        (*(SharedPtr<AlgorithmIface> *)addr);                                                   \
                result = cast(op(alg->call));                                                                   \
            } else if(method == deterministicCSR)                                                               \
            {                                                                                                   \
                SharedPtr<Batch<float, deterministicCSR> > alg =                                                \
                    staticPointerCast<Batch<float, deterministicCSR>, AlgorithmIface>                           \
                        (*(SharedPtr<AlgorithmIface> *)addr);                                                   \
                result = cast(op(alg->call));                                                                   \
            } else if(method == randomCSR)                                                                      \
            {                                                                                                   \
                SharedPtr<Batch<float, randomCSR> > alg =                                                       \
                    staticPointerCast<Batch<float, randomCSR>, AlgorithmIface>                                  \
                        (*(SharedPtr<AlgorithmIface> *)addr);                                                   \
                result = cast(op(alg->call));                                                                   \
            }                                                                                                   \
        }                                                                                                       \
    }                                                                                                           \
    else if(cmode == jDistributed)                                                                              \
    {                                                                                                           \
        if(step == jStep1Local)                                                                                 \
        {                                                                                                       \
            if(precision == 0)                                                                                  \
            {                                                                                                   \
                if(method == deterministicDense)                                                                \
                {                                                                                               \
                    SharedPtr<Distributed<step1Local, double, deterministicDense> > alg =                       \
                        staticPointerCast<Distributed<step1Local, double, deterministicDense>, AlgorithmIface>  \
                            (*(SharedPtr<AlgorithmIface> *)addr);                                               \
                    result = cast(op(alg->call));                                                               \
                } else if(method == randomDense)                                                                \
                {                                                                                               \
                    SharedPtr<Distributed<step1Local, double, randomDense> > alg =                              \
                        staticPointerCast<Distributed<step1Local, double, randomDense>, AlgorithmIface>         \
                            (*(SharedPtr<AlgorithmIface> *)addr);                                               \
                    result = cast(op(alg->call));                                                               \
                } else if(method == deterministicCSR)                                                           \
                {                                                                                               \
                    SharedPtr<Distributed<step1Local, double, deterministicCSR> > alg =                         \
                        staticPointerCast<Distributed<step1Local, double, deterministicCSR>, AlgorithmIface>    \
                            (*(SharedPtr<AlgorithmIface> *)addr);                                               \
                    result = cast(op(alg->call));                                                               \
                } else if(method == randomCSR)                                                                  \
                {                                                                                               \
                    SharedPtr<Distributed<step1Local, double, randomCSR> > alg =                                \
                        staticPointerCast<Distributed<step1Local, double, randomCSR>, AlgorithmIface>           \
                            (*(SharedPtr<AlgorithmIface> *)addr);                                               \
                    result = cast(op(alg->call));                                                               \
                }                                                                                               \
            }                                                                                                   \
            else                                                                                                \
            {                                                                                                   \
                if(method == deterministicDense)                                                                \
                {                                                                                               \
                    SharedPtr<Distributed<step1Local, float, deterministicDense> > alg =                        \
                        staticPointerCast<Distributed<step1Local, float, deterministicDense>, AlgorithmIface>   \
                            (*(SharedPtr<AlgorithmIface> *)addr);                                               \
                    result = cast(op(alg->call));                                                               \
                } else if(method == randomDense)                                                                \
                {                                                                                               \
                    SharedPtr<Distributed<step1Local, float, randomDense> > alg =                               \
                        staticPointerCast<Distributed<step1Local, float, randomDense>, AlgorithmIface>          \
                            (*(SharedPtr<AlgorithmIface> *)addr);                                               \
                    result = cast(op(alg->call));                                                               \
                } else if(method == deterministicCSR)                                                           \
                {                                                                                               \
                    SharedPtr<Distributed<step1Local, float, deterministicCSR> > alg =                          \
                        staticPointerCast<Distributed<step1Local, float, deterministicCSR>, AlgorithmIface>     \
                            (*(SharedPtr<AlgorithmIface> *)addr);                                               \
                    result = cast(op(alg->call));                                                               \
                } else if(method == randomCSR)                                                                  \
                {                                                                                               \
                    SharedPtr<Distributed<step1Local, float, randomCSR> > alg =                                 \
                        staticPointerCast<Distributed<step1Local, float, randomCSR>, AlgorithmIface>            \
                            (*(SharedPtr<AlgorithmIface> *)addr);                                               \
                    result = cast(op(alg->call));                                                               \
                }                                                                                               \
            }                                                                                                   \
        }                                                                                                       \
        else if(step == jStep2Master)                                                                           \
        {                                                                                                       \
            if(precision == 0)                                                                                  \
            {                                                                                                   \
                if(method == deterministicDense)                                                                \
                {                                                                                               \
                    SharedPtr<Distributed<step2Master, double, deterministicDense> > alg =                      \
                        staticPointerCast<Distributed<step2Master, double, deterministicDense>, AlgorithmIface> \
                            (*(SharedPtr<AlgorithmIface> *)addr);                                               \
                    result = cast(op(alg->call));                                                               \
                } else if(method == randomDense)                                                                \
                {                                                                                               \
                    SharedPtr<Distributed<step2Master, double, randomDense> > alg =                             \
                        staticPointerCast<Distributed<step2Master, double, randomDense>, AlgorithmIface>        \
                            (*(SharedPtr<AlgorithmIface> *)addr);                                               \
                    result = cast(op(alg->call));                                                               \
                } else if(method == deterministicCSR)                                                           \
                {                                                                                               \
                    SharedPtr<Distributed<step2Master, double, deterministicCSR> > alg =                        \
                        staticPointerCast<Distributed<step2Master, double, deterministicCSR>, AlgorithmIface>   \
                            (*(SharedPtr<AlgorithmIface> *)addr);                                               \
                    result = cast(op(alg->call));                                                               \
                } else if(method == randomCSR)                                                                  \
                {                                                                                               \
                    SharedPtr<Distributed<step2Master, double, randomCSR> > alg =                               \
                        staticPointerCast<Distributed<step2Master, double, randomCSR>, AlgorithmIface>          \
                            (*(SharedPtr<AlgorithmIface> *)addr);                                               \
                    result = cast(op(alg->call));                                                               \
                }                                                                                               \
            }                                                                                                   \
            else                                                                                                \
            {                                                                                                   \
                if(method == deterministicDense)                                                                \
                {                                                                                               \
                    SharedPtr<Distributed<step2Master, float, deterministicDense> > alg =                       \
                        staticPointerCast<Distributed<step2Master, float, deterministicDense>, AlgorithmIface>  \
                            (*(SharedPtr<AlgorithmIface> *)addr);                                               \
                    result = cast(op(alg->call));                                                               \
                } else if(method == randomDense)                                                                \
                {                                                                                               \
                    SharedPtr<Distributed<step2Master, float, randomDense> > alg =                              \
                        staticPointerCast<Distributed<step2Master, float, randomDense>, AlgorithmIface>         \
                            (*(SharedPtr<AlgorithmIface> *)addr);                                               \
                    result = cast(op(alg->call));                                                               \
                } else if(method == deterministicCSR)                                                           \
                {                                                                                               \
                    SharedPtr<Distributed<step2Master, float, deterministicCSR> > alg =                         \
                        staticPointerCast<Distributed<step2Master, float, deterministicCSR>, AlgorithmIface>    \
                            (*(SharedPtr<AlgorithmIface> *)addr);                                               \
                    result = cast(op(alg->call));                                                               \
                } else if(method == randomCSR)                                                                  \
                {                                                                                               \
                    SharedPtr<Distributed<step2Master, float, randomCSR> > alg =                                \
                        staticPointerCast<Distributed<step2Master, float, randomCSR>, AlgorithmIface>           \
                            (*(SharedPtr<AlgorithmIface> *)addr);                                               \
                    result = cast(op(alg->call));                                                               \
                }                                                                                               \
            }                                                                                                   \
        }                                                                                                       \
    }
