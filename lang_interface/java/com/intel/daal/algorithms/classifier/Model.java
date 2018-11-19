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
 * @defgroup classifier Base Classifier
 * @brief Contains classes for working with classifiers
 * @ingroup classification
 */
package com.intel.daal.algorithms.classifier;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__CLASSIFIER__MODEL"></a>
 * @brief Base class for models of the classification algorithms.
 */
public class Model extends com.intel.daal.algorithms.Model {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the model of the classification algorithm
     * @param context   Context to manage the model of the classification algorithm
     */
    public Model(DaalContext context) {
        super(context);
    }

    public Model(DaalContext context, long cModel) {
        super(context, cModel);
    }
}
/** @} */
