/* file: ModelNormEq.java */
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
 * @ingroup ridge_regression
 * @{
 */
package com.intel.daal.algorithms.ridge_regression;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__RIDGE_REGRESSION__MODELNORMEQ"></a>
 * @brief %Model trained by the ridge regression algorithm using the normal equations method
 *
 */
public class ModelNormEq extends Model {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public ModelNormEq(DaalContext context, long cModel) {
        super(context, cModel);
    }
}
/** @} */
