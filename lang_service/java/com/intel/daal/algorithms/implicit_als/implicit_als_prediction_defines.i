/* file: implicit_als_prediction_defines.i */
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
