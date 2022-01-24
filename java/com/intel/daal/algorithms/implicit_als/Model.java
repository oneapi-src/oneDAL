/* file: Model.java */
/*******************************************************************************
* Copyright 2014-2022 Intel Corporation
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

/**
 * @defgroup recommendation_systems Recommendation Systems
 * @ingroup training_and_prediction
 * @{
 */
/**
 * @defgroup implicit_als Implicit Alternating Least Squares
 * @brief Contains classes of the implicit ALS algorithm
 * @ingroup recommendation_systems
 * @{
 */
package com.intel.daal.algorithms.implicit_als;

import com.intel.daal.utils.*;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__MODEL"></a>
 * @brief Base class for the model trained by the implicit ALS algorithm in the batch processing mode
 *
 * @par References
 *      - Parameter class
 */
public class Model extends com.intel.daal.algorithms.Model {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public Model(DaalContext context, long cModel) {
        super(context, cModel);
    }

    /**
     * Returns the numeric table containing users factors constructed during the training of the implicit ALS model
     * @return Numeric table containing users factors
     */
    public NumericTable getUsersFactors() {
        return (NumericTable)Factory.instance().createObject(getContext(), cGetUsersFactors(this.getCObject()));
    }

    /**
     * Returns the numeric table containing items factors constructed during the training of the implicit ALS model
     * @return Numeric table containing items factors
     */
    public NumericTable getItemsFactors() {
        return (NumericTable)Factory.instance().createObject(getContext(), cGetItemsFactors(this.getCObject()));
    }

    protected native long cGetUsersFactors(long modelAddr);

    protected native long cGetItemsFactors(long modelAddr);
}
/** @} */
/** @} */
