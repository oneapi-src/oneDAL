/* file: Model.java */
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
