/* file: classifier_training_distributed.h */
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
//  Implementation of the base classes used in the training stage
//  of the classification algorithm
//--
*/

#ifndef __CLASSIFIER_TRAINING_DISTRIBUTED_H__
#define __CLASSIFIER_TRAINING_DISTRIBUTED_H__

#include "algorithms/algorithm.h"
#include "algorithms/classifier/classifier_training_types.h"

namespace daal
{
namespace algorithms
{
namespace classifier
{
namespace training
{

namespace interface1
{
/**
 * @defgroup classifier_training_distributed Distributed
 * @ingroup training
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__CLASSIFIER__TRAINING__DISTRIBUTED"></a>
 *  \brief Algorithm class for training the classifier model in the distributed processing mode
 *
 * \par Enumerations
 *      - \ref InputId  Identifiers of input objects of the classifier model training algorithm
 *      - \ref ResultId Identifiers of results of the classifier model training algorithm
 *
 * \par References
 *      - \ref interface1::Parameter "Parameter" class
 *      - DistributedInput class
 *      - \ref interface1::Model "Model" class
 *      - Result class
 */
class Distributed : public Training<distributed>
{
public:
    DistributedInput input;     /*!< Input objects for the classifier model training algorithm in the distributed processing mode */

    Distributed()
    {
        initialize();
    }

    /**
     * Constructs a classifier training algorithm by copying input objects and parameters
     * of another classifier training algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Distributed(const Distributed &other)
    {
        initialize();
        this->input.set(partialModels,    other.input.get(partialModels));
    }

    virtual ~Distributed() {}

    /**
     * Registers user-allocated memory for storing partial training results
     * \param[in] partialResult    Structure for storing partial results
     */
    void setPartialResult(const services::SharedPtr<PartialResult>& partialResult)
    {
        _partialResult = partialResult;
        _pres = _partialResult.get();
    }

    /**
     * Registers user-allocated memory for storing results of the classification algorithm
     * \param[in] res    Structure for storing results of the classification algorithm
     */
    void setResult(const services::SharedPtr<Result>& res)
    {
        DAAL_CHECK(res, ErrorNullResult)
        _result = res;
        _res = _result.get();
    }

    /**
     * Returns the structure that contains computed partial results
     * \return Structure that contains computed partial results
     */
    services::SharedPtr<PartialResult> getPartialResult() { return _partialResult; }

    /**
     * Returns the structure that contains computed results of the classification algorithm
     * \return Structure that contains computed results of the classification algorithm
     */
    services::SharedPtr<Result> getResult() { return _result; }

    /**
     * Returns a pointer to the newly allocated classifier training algorithm with a copy of input objects
     * and parameters of this classifier training algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Distributed> clone() const
    {
        return services::SharedPtr<Distributed>(cloneImpl());
    }

protected:
    services::SharedPtr<PartialResult> _partialResult;
    services::SharedPtr<Result> _result;

    void initialize()
    {
        _in = &input;
    }
    virtual Distributed * cloneImpl() const DAAL_C11_OVERRIDE = 0;
};
/** @} */
} // namespace interface1
using interface1::DistributedInput;
using interface1::Distributed;

}
}
}
}
#endif
