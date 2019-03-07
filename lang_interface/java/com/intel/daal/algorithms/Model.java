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
 * @ingroup base_algorithms
 * @{
 */
package com.intel.daal.algorithms;

import com.intel.daal.data_management.data.SerializableBase;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MODEL"></a>
 * @brief Model is the base class for the classes that represent the models, such as
 * linear regression or Support Vector Machine classifier.
 */
abstract public class Model extends SerializableBase {

    /**
     * Constructs the model for the algorithm
     * @param context   Context to manage the model for the algorithm
     */
    protected Model(DaalContext context) {
        super(context);
    }

    /**
     * @brief Construct model from C++ model
     * @param context Context to manage the model
     * @param cModel  pointer to C++ model
     */
    public Model(DaalContext context, long cModel) {
        super(context);
        this.cObject = cModel;
    }
}
/** @} */
