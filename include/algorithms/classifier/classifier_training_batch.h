/* file: classifier_training_batch.h */
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
//  Implementation of the interface for the classifier model training algorithm.
//--
*/

#ifndef __CLASSIFIER_TRAINING_BATCH_H__
#define __CLASSIFIER_TRAINING_BATCH_H__

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
 * @defgroup classifier_training_batch Batch
 * @ingroup training
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__CLASSIFIER__TRAINING__BATCH"></a>
 * \brief Algorithm class for training the classifier model
 *
 * \par Enumerations
 *      - \ref InputId  Identifiers of input objects of the classifier model training algorithm
 *      - \ref ResultId Identifiers of results of the classifier model training algorithm
 *
 * \par References
 *      - \ref interface1::Parameter "Parameter" class
 *      - Input class
 *      - \ref interface1::Model "Model" class
 *      - Result class
 */
class Batch : public Training<batch>
{
public:
    Input input;    /*!< %Input objects of the algorithm */

    Batch()
    {
        initialize();
    }

    /**
     * Constructs a classifier training algorithm by copying input objects and parameters
     * of another classifier training algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Batch(const Batch &other)
    {
        initialize();
        this->input.set(data,    other.input.get(data));
        this->input.set(labels,  other.input.get(labels));
        this->input.set(weights, other.input.get(weights));
    }

    virtual ~Batch()
    {}

    /**
     * Registers user-allocated memory for storing results of the classifier model training algorithm
     * \param[in] res    Structure for storing results of the classifier model training algorithm
     */
    void setResult(const services::SharedPtr<Result>& res)
    {
        DAAL_CHECK(res, ErrorNullResult)
        _result = res;
        _res = _result.get();
    }

    /**
     * Returns the structure that contains the trained classifier model
     * \return Structure that contains the trained classifier model
     */
    services::SharedPtr<Result> getResult() { return _result; }

    /**
     * Resets the results of the classifier model training algorithm
     */
    virtual void resetResult() = 0;

    /**
     * Returns a pointer to the newly allocated classifier training algorithm with a copy of input objects
     * and parameters of this classifier training algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch> clone() const
    {
        return services::SharedPtr<Batch>(cloneImpl());
    }

protected:

    void initialize()
    {
        _in = &input;
    }
    virtual Batch * cloneImpl() const DAAL_C11_OVERRIDE = 0;
    services::SharedPtr<Result> _result;
};
/** @} */
} // namespace interface1
using interface1::Batch;

}
}
}
}
#endif
