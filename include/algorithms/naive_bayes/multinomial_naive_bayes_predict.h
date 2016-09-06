/* file: multinomial_naive_bayes_predict.h */
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
//  Implementation of the interface for multinomial naive Bayes model-based prediction
//  in the batch processing mode
//--
*/

#ifndef __NAIVE_BAYES_PREDICT_H__
#define __NAIVE_BAYES_PREDICT_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "services/daal_defines.h"
#include "multinomial_naive_bayes_predict_types.h"
#include "algorithms/classifier/classifier_predict.h"

namespace daal
{
namespace algorithms
{
namespace multinomial_naive_bayes
{
namespace prediction
{

/**
 * \brief Contains version 1.0 of the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface
 */
namespace interface1
{
/**
 * @defgroup multinomial_naive_bayes_prediction_batch Batch
 * @ingroup multinomial_naive_bayes_prediction
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTINOMIAL_NAIVE_BAYES__PREDICTION__PREDICTIONCONTAINER"></a>
 * \brief Runs the prediction based on the multinomial naive Bayes model
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for prediction based on the multinomial naive Bayes model, double or float
 * \tparam method           Multinomial naive Bayes prediction method, \ref Method
 */
template<typename algorithmFPType, prediction::Method method, CpuType cpu>
class DAAL_EXPORT PredictionContainer : public PredictionContainerIface
{
public:
    /**
     * Constructs a container for multinomial naive Bayes model-based prediction with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    PredictionContainer(daal::services::Environment::env *daalEnv);
    /** Default destructor */
    ~PredictionContainer();
    /**
     * Computes the result of multinomial naive Bayes model-based prediction in the batch processing mode
     */
    void compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTINOMIAL_NAIVE_BAYES__PREDICTION__BATCH"></a>
 *  \brief Predicts the results of the multinomial naive Bayes classification
 *  \n<a href="DAAL-REF-MULTINOMNAIVEBAYES-ALGORITHM">Multinomial naive Bayes algorithm description and usage models</a>
 *
 *  \tparam algorithmFPType  Data type to use in intermediate computations for prediction based on the multinomial naive Bayes model, double or float
 *  \tparam method           Multinomial naive Bayes prediction method, \ref Method
 *
 *  \par Enumerations
 *      - \ref Method Multinomial naive Bayes prediction methods
 */
template<typename algorithmFPType = double, prediction::Method method = defaultDense>
class Batch : public classifier::prediction::Batch
{
public:
    Input input;                /*!< Input objects of the algorithm */
    Parameter parameter;        /*!< Parameters of the prediction algorithm */
    /**
     * Default constructor
     * \param nClasses  Number of classes
     */
    Batch(size_t nClasses) : parameter(nClasses)
    {
        initialize();
    }

    /**
     * Constructs multinomial naive Bayes prediction algorithm by copying input objects and parameters
     * of another multinomial naive Bayes prediction algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Batch(const Batch<algorithmFPType, method> &other) :
        classifier::prediction::Batch(other), parameter(other.parameter.nClasses)
    {
        initialize();
        parameter = other.parameter;
        this->input.set(classifier::prediction::data,  other.input.get(classifier::prediction::data));
        this->input.set(classifier::prediction::model, other.input.get(classifier::prediction::model));
    }

    virtual ~Batch() {}

    /**
    * Returns method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return(int)method; }

    /**
     * Returns a pointer to the newly allocated multinomial naive Bayes prediction algorithm
     * with a copy of input objects and parameters of this multinomial naive Bayes prediction algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl());
    }

protected:

    virtual Batch<algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Batch<algorithmFPType, method>(*this);
    }

    void allocateResult() DAAL_C11_OVERRIDE
    {
        _result->allocate<algorithmFPType>(inputBase, &parameter, (int) method);
        _res = _result.get();
    }

    void initialize()
    {
        inputBase = &input;
        _in = &input;
        _ac = new __DAAL_ALGORITHM_CONTAINER(batch, PredictionContainer, algorithmFPType, method)(&_env);
        _par = &parameter;
    }
};
/** @} */
} // namespace interface1
using interface1::PredictionContainer;
using interface1::Batch;

} // namespace prediction
} // namespace multinomial_naive_bayes
} // namespace algorithms
} // namespace daal
#endif
