/* file: pca_result_v1.h */
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

#ifndef __PCA_RESULT_V1_H__
#define __PCA_RESULT_V1_H__

#include "pca/inner/pca_types_v1.h"
#include "data_management/data/data_collection.h"
#include "data_management/data/numeric_table.h"


namespace daal
{
namespace algorithms
{
namespace pca
{
namespace interface1
{

class ResultImpl : public data_management::interface1::DataCollection
{
public:
    DAAL_CAST_OPERATOR(ResultImpl);

    ResultImpl(const size_t n);
    ResultImpl(const ResultImpl& o);
    virtual ~ResultImpl();

    /**
    * Allocates memory for storing partial results of the PCA algorithm
    * \param[in] input Pointer to an object containing input data
    * \return Status of computations
    */
    template <typename algorithmFPType>
    services::Status allocate(const daal::algorithms::Input *input);

    /**
    * Allocates memory for storing partial results of the PCA algorithm
    * \param[in] partialResult Pointer to an object containing partialResult data
    * \return Status of computations
    */
    template <typename algorithmFPType>
    services::Status allocate(const daal::algorithms::PartialResult *partialResult);

    /**
    * Checks the results of the PCA algorithm implementation
    * \param[in] nFeatures      Number of features
    * \param[in] nTables        Number of tables
    *
    * \return Status
    */
    virtual services::Status check(size_t nFeatures, size_t nTables) const;

    /**
    * Sets numeric table to storage
    * \param[in] key     key
    * \param[in] table   table
    */
    virtual void setTable(size_t key, data_management::NumericTablePtr table);

protected:

    /**
    * Allocates memory for storing partial results of the PCA algorithm
    * \param[in] nFeatures Number of features
    * \return Status of computations
    */
    template <typename algorithmFPType>
    services::Status allocate(size_t nFeatures);

};

} // interface1
} // namespace pca
} // namespace algorithms
} // namespace daal

#endif
