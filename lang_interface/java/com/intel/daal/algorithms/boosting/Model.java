/* file: Model.java */
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

/**
 * @defgroup boosting Boosting
 * @ingroup classification
 * @{
 */
package com.intel.daal.algorithms.boosting;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__BOOSTING__MODEL"></a>
 * @brief Base class for models of %boosting algorithms.
 *        Contains a collection of weak learner algorithm models constructed during training of the %boosting algorithm
 */
public class Model extends com.intel.daal.algorithms.classifier.Model {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the model
     * @param context  Context to manage the model
     */
    public Model(DaalContext context) {
        super(context);
    }

    public Model(DaalContext context, long cModel) {
        super(context, cModel);
    }
}
/** @} */
