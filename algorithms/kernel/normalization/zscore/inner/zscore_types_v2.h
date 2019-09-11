/* file: zscore_types_v2.h */
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

/*
//++
//  Definition of common types of z-score normalization.
//--
*/

#ifndef __ZSCORE_TYPES_V2_H__
#define __ZSCORE_TYPES_V2_H__

#include "zscore_types.h"
#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"
#include "algorithms/moments/low_order_moments_batch.h"
#include "services/daal_defines.h"
#include "algorithms/algorithm.h"

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

/**
 * \brief Contains version 2.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface2
{

/**
* <a name="DAAL-CLASS-ALGORITHMS__NORMALIZATION__ZSCORE__PARAMETER"></a>
* \brief Class that specifies the parameters of the algorithm in the batch computing mode
*/
template<typename algorithmFPType, Method method>
class DAAL_EXPORT Parameter {};

/**
* <a name="DAAL-CLASS-ALGORITHMS__NORMALIZATION__ZSCORE__BASEPARAMETER"></a>
* \brief Class that specifies the base parameters of the algorithm in the batch computing mode
*/
class DAAL_EXPORT BaseParameter : public daal::algorithms::Parameter
{
public:
    BaseParameter();
    DAAL_UINT64 resultsToCompute;  /*!< 64 bit integer flag that indicates the results to compute */
};

// /**
//  * <a name="DAAL-CLASS-ALGORITHMS__NORMALIZATION__ZSCORE__PARAMETER"></a>
//  * \brief Class that specifies the parameters of the default algorithm in the batch computing mode
//  */
template<typename algorithmFPType>
class DAAL_EXPORT Parameter<algorithmFPType, sumDense> : public BaseParameter
{
public:
    Parameter();
};

// /**
//  * <a name="DAAL-CLASS-ALGORITHMS__NORMALIZATION__ZSCORE__PARAMETER"></a>
//  * \brief Class that specifies the parameters of the default algorithm in the batch computing mode
//  */
template<typename algorithmFPType>
class DAAL_EXPORT Parameter<algorithmFPType, defaultDense> : public BaseParameter
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

/** @} */
/** @} */
} // namespace interface2

} // namespace zscore
} // namespace normalization
} // namespace algorithms
} // namespace daal
#endif // __ZSCORE_TYPES_V2_H__
