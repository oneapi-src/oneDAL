/* file: kdtree_knn_classification_predict.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
//  Implementation of the interface for K-Nearest Neighbors (kNN) model-based prediction
//--
*/

#ifndef __KDTREE_KNN_CLASSIFICATION_PREDICT_H__
#define __KDTREE_KNN_CLASSIFICATION_PREDICT_H__

#include "algorithms/algorithm.h"
#include "algorithms/k_nearest_neighbors/kdtree_knn_classification_predict_types.h"
#include "algorithms/k_nearest_neighbors/kdtree_knn_classification_model.h"
#include "algorithms/classifier/classifier_predict.h"
#include "data_management/data/homogen_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace kdtree_knn_classification
{
namespace prediction
{
namespace interface1
{
/**
 * @defgroup kdtree_knn_classification_prediction_batch Batch
 * @ingroup kdtree_knn_classification_prediction
 * @{
 */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__PREDICTION__BATCHCONTAINER"></a>
 *  \brief Class containing computation methods for KD-tree based kNN model-based prediction   \DAAL_DEPRECATED
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public PredictionContainerIface
{
public:
    /**
     * Constructs a container for KD-tree based kNN model-based prediction with a specified environment
     * \param[in] daalEnv   Environment object
     */
    DAAL_DEPRECATED BatchContainer(daal::services::Environment::env * daalEnv);

    DAAL_DEPRECATED ~BatchContainer();

    /**
     *  Computes the result of KD-tree based kNN model-based prediction
     */
    DAAL_DEPRECATED services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__PREDICTION__BATCH"></a>
 * \brief Provides methods to run implementations of the KD-tree based kNN model-based prediction
 * <!-- \n<a href="DAAL-REF-KNN-ALGORITHM">kNN algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for KD-tree based kNN model-based prediction
 *                          in the batch processing mode, double or float
 * \tparam method           Computation method in the batch processing mode, \ref Method
 *
 * \par Enumerations
 *      - \ref Method  Computation methods for KD-tree based kNN model-based prediction
 *
 * \par References
 *      - \ref kdtree_knn_classification::interface1::Model "kdtree_knn_classification::Model" class
 *      - \ref training::interface1::Batch "training::Batch" class
 */
template <typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class Batch : public classifier::prediction::interface1::Batch
{
public:
    typedef classifier::prediction::interface1::Batch super;

    typedef algorithms::kdtree_knn_classification::prediction::Input InputType;
    typedef algorithms::kdtree_knn_classification::interface1::Parameter ParameterType;
    typedef typename super::ResultType ResultType;

    InputType input;         /*!< %Input data structure */
    ParameterType parameter; /*!< \ref kdtree_knn_classification::interface1::Parameter "Parameters" of prediction */

    /** Default constructor */
    DAAL_DEPRECATED Batch() { initialize(); }

    /**
     * Constructs a KD-tree based kNN prediction algorithm by copying input objects and parameters
     * of another KD-tree based kNN prediction algorithm
     * \param[in] other Algorithm to use as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    DAAL_DEPRECATED Batch(const Batch<algorithmFPType, method> & other)
        : classifier::prediction::interface1::Batch(other), input(other.input), parameter(other.parameter)
    {
        initialize();
    }

    /**
     * Get input objects for the KD-tree based kNN prediction algorithm
     * \return %Input objects for the KD-tree based kNN prediction algorithm
     */
    DAAL_DEPRECATED InputType * getInput() DAAL_C11_OVERRIDE { return &input; }

    /**
     * Returns the method of the algorithm
     * \return Method of the algorithm
     */
    DAAL_DEPRECATED_VIRTUAL virtual int getMethod() const DAAL_C11_OVERRIDE { return (int)method; }

    /**
     * Returns a pointer to the newly allocated KD-tree based kNN prediction algorithm with a copy of input objects
     * of this KD-tree based kNN prediction algorithm
     * \return Pointer to the newly allocated algorithm
     */
    DAAL_DEPRECATED services::SharedPtr<Batch<algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl());
    }

protected:
    virtual Batch<algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE { return new Batch<algorithmFPType, method>(*this); }

    services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        services::Status s = _result->allocate<algorithmFPType>(&input, &parameter, (int)method);
        _res               = _result.get();
        return s;
    }

    void initialize()
    {
        _in  = &input;
        _ac  = new __DAAL_ALGORITHM_CONTAINER(batch, BatchContainer, algorithmFPType, method)(&_env);
        _par = &parameter;
    }
};

/** @} */
} // namespace interface1

namespace interface2
{
/**
 * @defgroup kdtree_knn_classification_prediction_batch Batch
 * @ingroup kdtree_knn_classification_prediction
 * @{
 */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__PREDICTION__BATCHCONTAINER"></a>
 *  \brief Class containing computation methods for KD-tree based kNN model-based prediction
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public PredictionContainerIface
{
public:
    /**
     * Constructs a container for KD-tree based kNN model-based prediction with a specified environment
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);

    ~BatchContainer();

    /**
     *  Computes the result of KD-tree based kNN model-based prediction
     */
    services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__PREDICTION__BATCH"></a>
 * \brief Provides methods to run implementations of the KD-tree based kNN model-based prediction
 * <!-- \n<a href="DAAL-REF-KNN-ALGORITHM">kNN algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for KD-tree based kNN model-based prediction
 *                          in the batch processing mode, double or float
 * \tparam method           Computation method in the batch processing mode, \ref Method
 *
 * \par Enumerations
 *      - \ref Method  Computation methods for KD-tree based kNN model-based prediction
 *
 * \par References
 *      - \ref kdtree_knn_classification::interface1::Model "kdtree_knn_classification::Model" class
 *      - \ref training::interface1::Batch "training::Batch" class
 */
template <typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class Batch : public classifier::prediction::Batch
{
public:
    typedef classifier::prediction::Batch super;

    typedef algorithms::kdtree_knn_classification::prediction::Input InputType;
    typedef algorithms::kdtree_knn_classification::Parameter ParameterType;
    typedef typename super::ResultType ResultType;

    InputType input;         /*!< %Input data structure */
    ParameterType parameter; /*!< \ref kdtree_knn_classification::interface1::Parameter "Parameters" of prediction */

    /** Default constructor */
    Batch() { initialize(); }

    /**
     * Constructs a KD-tree based kNN prediction algorithm by copying input objects and parameters
     * of another KD-tree based kNN prediction algorithm
     * \param[in] other Algorithm to use as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Batch(const Batch<algorithmFPType, method> & other) : classifier::prediction::Batch(other), input(other.input), parameter(other.parameter)
    {
        initialize();
    }

    /**
     * Get input objects for the KD-tree based kNN prediction algorithm
     * \return %Input objects for the KD-tree based kNN prediction algorithm
     */
    InputType * getInput() DAAL_C11_OVERRIDE { return &input; }

    /**
     * Returns the method of the algorithm
     * \return Method of the algorithm
     */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return (int)method; }

    /**
     * Returns a pointer to the newly allocated KD-tree based kNN prediction algorithm with a copy of input objects
     * of this KD-tree based kNN prediction algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const { return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl()); }

protected:
    virtual Batch<algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE { return new Batch<algorithmFPType, method>(*this); }

    services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        services::Status s = _result->allocate<algorithmFPType>(&input, &parameter, (int)method);
        _res               = _result.get();
        return s;
    }

    void initialize()
    {
        _in  = &input;
        _ac  = new __DAAL_ALGORITHM_CONTAINER(batch, BatchContainer, algorithmFPType, method)(&_env);
        _par = &parameter;
    }
};

/** @} */
} // namespace interface2

using interface2::BatchContainer;
using interface2::Batch;

} // namespace prediction
} // namespace kdtree_knn_classification
} // namespace algorithms
} // namespace daal

#endif
