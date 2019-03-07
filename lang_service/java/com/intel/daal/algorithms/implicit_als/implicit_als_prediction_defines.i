/* file: implicit_als_prediction_defines.i */
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

#include "implicit_als/prediction/ratings/JRatingsPartialModelInputId.h"
#include "implicit_als/prediction/ratings/JRatingsModelInputId.h"
#include "implicit_als/prediction/ratings/JRatingsPartialResultId.h"
#include "implicit_als/prediction/ratings/JRatingsResultId.h"
#include "implicit_als/prediction/ratings/JRatingsMethod.h"

#include "common_defines.i"

#define defaultDenseId          com_intel_daal_algorithms_implicit_als_prediction_ratings_RatingsMethod_defaultDenseId

#define modelId                 com_intel_daal_algorithms_implicit_als_prediction_ratings_RatingsModelInputId_modelId

#define usersPartialModelId     com_intel_daal_algorithms_implicit_als_prediction_ratings_RatingsPartialModelInputId_usersPartialModelId
#define itemsPartialModelId     com_intel_daal_algorithms_implicit_als_prediction_ratings_RatingsPartialModelInputId_itemsPartialModelId

#define predictionId            com_intel_daal_algorithms_implicit_als_prediction_ratings_RatingsResultId_predictionId

#define finalResultId           com_intel_daal_algorithms_implicit_als_prediction_ratings_RatingsPartialResultId_finalResultId
