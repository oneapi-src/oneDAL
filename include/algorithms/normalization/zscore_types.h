/* file: zscore_types.h */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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
//  Definition of common types of z-score normalization.
//--
*/

#ifndef __ZSCORE_TYPES_H__
#define __ZSCORE_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"
#include "algorithms/moments/low_order_moments_batch.h"
#include "services/daal_defines.h"

namespace daal
{
namespace algorithms
{
/**
 * @defgroup normalization Normalization
 * \copydoc daal::algorithms::normalization
 * @ingroup analysis
 * @{
 */
/**
 * \brief Contains classes to run the z-score normalization algorithms
 */
namespace normalization
{
/**
 * @defgroup zscore Z-score
 * \copydoc daal::algorithms::normalization::zscore
 * @ingroup normalization
 * @{
 */
/**
* \brief Contains classes for computing the z-score normalization
*/
namespace zscore
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__NORMALIZATION__ZSCORE__METHOD"></a>
 * Available methods for z-score normalization computation
 * @ingroup zscore
 */
enum Method
{
    defaultDense = 0,      /*!< Default: performance-oriented method. Works with all types of numeric tables */
    sumDense     = 1,      /*!< Precomputed sum: implementation of algorithm in the case of a precomputed sum.
                                     Works with all types of numeric tables */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__NORMALIZATION__ZSCORE__INPUTID"></a>
 * Available identifiers of input objects for the z-score normalization algorithm
 * @ingroup zscore
 */
enum InputId
{
    data,             /*!< %Input data table */
    lastInputId = data
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__NORMALIZATION__ZSCORE__RESULTID"></a>
 * Available identifiers of results of the z-score normalization algorithm
 * @ingroup zscore
 */
enum ResultId
{
    normalizedData,        /*!< z-score normalization results */
    lastResultId = normalizedData
};

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
* <a name="DAAL-CLASS-ALGORITHMS__NORMALIZATION__ZSCORE__PARAMETER"></a>
* \brief Class that specifies the parameters of the algorithm in the batch computing mode
*/
template<typename algorithmFPType, Method method>
class DAAL_EXPORT Parameter : public daal::algorithms::Parameter {};

// /**
//  * <a name="DAAL-CLASS-ALGORITHMS__NORMALIZATION__ZSCORE__PARAMETER"></a>
//  * \brief Class that specifies the parameters of the default algorithm in the batch computing mode
//  */
template<typename algorithmFPType>
class DAAL_EXPORT Parameter<algorithmFPType, defaultDense> : public daal::algorithms::Parameter
{
public:
    /** Constructs z-score normalization parameters */
    Parameter(const services::SharedPtr<low_order_moments::BatchImpl> &momentsForParameter =
                  services::SharedPtr<low_order_moments::Batch<algorithmFPType, low_order_moments::defaultDense> >
              (new low_order_moments::Batch<algorithmFPType, low_order_moments::defaultDense>()));

    services::SharedPtr<low_order_moments::BatchImpl> moments; /*!< Pointer to the algorithm that computes the low order moments */

    /**
     * Check the correctness of the %Parameter object
     *
     * \return Status of computations
     */
    virtual services::Status check() const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NORMALIZATION__ZSCORE__INPUT"></a>
 * \brief %Input objects for the z-score normalization algorithm
 */
class DAAL_EXPORT Input : public daal::algorithms::Input
{
public:
    /** Default constructor */
    Input();

    /** Copy constructor */
    Input(const Input& other);

    virtual ~Input() {}

    /**
     * Returns an input object for the z-score normalization algorithm
     * \param[in] id    Identifier of the %input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(InputId id) const;

    /**
     * Sets the input object of the z-score normalization algorithm
     * \param[in] id    Identifier of the %input object
     * \param[in] ptr   Pointer to the input object
     */
    void set(InputId id, const data_management::NumericTablePtr &ptr);

    /**
     * Check the correctness of the %Input object
     * \param[in] par       Algorithm parameter
     * \param[in] method    Algorithm computation method
     *
     * \return Status of computations
     */
    services::Status check(const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NORMALIZATION__ZSCORE__RESULT"></a>
 * \brief Provides methods to access final results obtained with the compute() method of the
 *        z-score normalization algorithm in the batch processing mode
 */
class DAAL_EXPORT Result : public daal::algorithms::Result
{
public:
    DECLARE_SERIALIZABLE();
    Result();

    virtual ~Result() {};

    /**
     * Allocates memory to store final results of the z-score normalization algorithms
     * \param[in] input     Input objects for the z-score normalization algorithm
     * \param[in] method    Algorithm computation method
     *
     * \return Status of computations
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input *input, const int method);

    /**
     * Returns the final result of the z-score normalization algorithm
     * \param[in] id   Identifier of the final result, daal::algorithms::normalization::zscore::ResultId
     * \return         Final result that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(ResultId id) const;

    /**
     * Sets the Result object of the z-score normalization algorithm
     * \param[in] id        Identifier of the Result object
     * \param[in] value     Pointer to the Result object
     */
    void set(ResultId id, const data_management::NumericTablePtr &value);

    /**
     * Checks the correctness of the Result object
     * \param[in] in     Pointer to the input object
     * \param[in] par    Pointer to the parameter object
     * \param[in] method Algorithm computation method
     *
     * \return Status of computations
     */
    services::Status check(const daal::algorithms::Input *in, const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;

protected:
    /** \private */
    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch)
    {
        daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);
    }

    void serializeImpl(data_management::InputDataArchive  *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::InputDataArchive, false>(arch);}

    void deserializeImpl(data_management::OutputDataArchive *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::OutputDataArchive, true>(arch);}
};
typedef services::SharedPtr<Result> ResultPtr;

/** @} */
/** @} */
} // namespace interface1
using interface1::Parameter;
using interface1::Input;
using interface1::Result;
using interface1::ResultPtr;

} // namespace zscore
} // namespace normalization
} // namespace algorithms
} // namespace daal
#endif
