/* file: pca_result_impl.h */
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
//  Implementation of PCA algorithm result.
//--
*/

#ifndef __PCA_RESULT_IMPL_H__
#define __PCA_RESULT_IMPL_H__

#include "inner/pca_result_v1.h"

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace interface2
{

class ResultImpl : public interface1::ResultImpl
{
public:
    DAAL_CAST_OPERATOR(ResultImpl);

    bool isWhitening;
    ResultImpl(const size_t n) : interface1::ResultImpl(n), isWhitening(false) {}
    ResultImpl(const ResultImpl& o) : interface1::ResultImpl(o), isWhitening(o.isWhitening){}
    virtual ~ResultImpl() {};

    /**
    * Allocates memory for storing partial results of the PCA algorithm
    * \param[in] input Pointer to an object containing input data
    * \param[in] nComponents Number of components
    * \param[in] resultsToCompute Results to compute
    * \return Status of computations
    */
    template <typename algorithmFPType>
    services::Status allocate(const daal::algorithms::Input *input, size_t nComponents, DAAL_UINT64 resultsToCompute);

    /**
    * Allocates memory for storing partial results of the PCA algorithm
    * \param[in] partialResult Pointer to an object containing partialResult data
    * \param[in] nComponents Number of components
    * \param[in] resultsToCompute Results to compute
    * \return Status of computations
    */
    template <typename algorithmFPType>
    services::Status allocate(const daal::algorithms::PartialResult *partialResult, size_t nComponents, DAAL_UINT64 resultsToCompute);

    /**
    * Checks the results of the PCA algorithm implementation
    * \param[in] nFeatures      Number of features
    * \param[in] nComponents Number of components
    * \param[in] nTables        Number of tables
    *
    * \return Status
    */
    virtual services::Status check(size_t nFeatures, size_t nComponents, size_t nTables) const;

protected:

    /**
    * Allocates memory for storing partial results of the PCA algorithm
    * \param[in] nFeatures Number of features
    * \param[in] nComponents Number of components
    * \param[in] resultsToCompute Results to compute
    * \return Status of computations
    */
    template <typename algorithmFPType>
    services::Status allocate(size_t nFeatures, size_t nComponents, DAAL_UINT64 resultsToCompute);
};

} // interface1
} // namespace pca
} // namespace algorithms
} // namespace daal

#endif
