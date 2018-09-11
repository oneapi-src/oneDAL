/* file: SplitParameter.java */
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
 * @ingroup split
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.split;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__SPLIT__SPLITPARAMETER"></a>
 * \brief Class that specifies parameters of the split layer
 */
public class SplitParameter extends com.intel.daal.algorithms.neural_networks.layers.Parameter {

    /**
     *  Constructs the parameters for the split layer
     */
    public SplitParameter(DaalContext context) {
        super(context);
        cObject = cInit();
    }

    public SplitParameter(DaalContext context, long cParameter) {
        super(context, cParameter);
    }

    /**
     *  Gets the number of outputs for forward split layer
     */
    public double getNOutputs() {
        return cGetNOutputs(cObject);
    }

    /**
     *  Sets the number of outputs for forward split layer
     *  @param nOutputs Number of outputs for forward split layer
     */
    public void setNOutputs(long nOutputs) {
        cSetNOutputs(cObject, nOutputs);
    }

    /**
     *  Gets the number of inputs for backward split layer
     */
    public long getNInputs() {
        return cGetNInputs(cObject);
    }

    /**
     *  Sets the number of inputs for backward split layer
     *  @param nInputs Number of inputs for backward split layer
     */
    public void setNInputs(long nInputs) {
        cSetNInputs(cObject, nInputs);
    }

    private native long cInit();
    private native long cGetNOutputs(long cParameter);
    private native void cSetNOutputs(long cParameter, long nOutputs);
    private native long cGetNInputs(long cParameter);
    private native void cSetNInputs(long cParameter, long nInputs);
}
/** @} */
