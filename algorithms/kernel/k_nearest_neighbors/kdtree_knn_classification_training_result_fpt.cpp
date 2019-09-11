/* file: kdtree_knn_classification_training_result_fpt.cpp */
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
//  Implementation of the class defining the K-Nearest Neighbors (kNN) model
//--
*/

#include "kdtree_knn_classification_training_result.h"

namespace daal
{
namespace algorithms
{
namespace kdtree_knn_classification
{
namespace training
{

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input * input, const Parameter * parameter, int method);

}// namespace training
}// namespace kdtree_knn_classification
}// namespace algorithms
}// namespace daal
