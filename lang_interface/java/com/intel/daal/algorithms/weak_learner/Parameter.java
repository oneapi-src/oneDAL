/* file: Parameter.java */
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
 * @ingroup weak_learner
 * @{
 */
package com.intel.daal.algorithms.weak_learner;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__WEAK_LEARNER__PARAMETER"></a>
 * @brief Base class for the input objects of the weak learner training and prediction algorithm
 */
public class Parameter extends com.intel.daal.algorithms.classifier.Parameter {

    /**
     * Constructs the parameter of the weak learner algorithm
     * @param context   Context to manage the parameter of the weak learner algorithm
     */
    public Parameter(DaalContext context) {
        super(context);
    }

    public Parameter(DaalContext context, long cParameter) {
        super(context, cParameter);
    }
}
/** @} */
