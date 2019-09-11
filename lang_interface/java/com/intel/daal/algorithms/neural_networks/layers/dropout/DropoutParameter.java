/* file: DropoutParameter.java */
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
 * @ingroup dropout
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.dropout;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__DROPOUT__DROPOUTPARAMETER"></a>
 * \brief Class that specifies parameters of the dropout layer
 */
public class DropoutParameter extends com.intel.daal.algorithms.neural_networks.layers.Parameter {

    /**
     *  Constructs the parameters for the dropout layer
     * @param context Context to manage the parameter for the dropout layer
     */
    public DropoutParameter(DaalContext context) {
        super(context);
        cObject = cInit();
    }

    public DropoutParameter(DaalContext context, long cParameter) {
        super(context, cParameter);
    }

    /**
     *  Gets the probability that any particular element is retained
     */
    public double getRetainRatio() {
        return cGetRetainRatio(cObject);
    }

    /**
     *  Sets the probability that any particular element is retained
     *  @param retainRatio Probability that any particular element is retained
     */
    public void setRetainRatio(double retainRatio) {
        cSetRetainRatio(cObject, retainRatio);
    }

    /**
     * @DAAL_DEPRECATED
     *  Gets the seed for mask elements random generation
     */
    public long getSeed() {
        return cGetSeed(cObject);
    }

    /**
     * @DAAL_DEPRECATED
     *  Sets the seed for mask elements random generation
     *  @param seed Seed for mask elements random generation
     */
    public void setSeed(long seed) {
       cSetSeed(cObject, seed);
    }

    /**
     * Sets the engine for mask elements random generation
     * @param engine for mask elements random generation
     */
    public void setEngine(com.intel.daal.algorithms.engines.BatchBase engine) {
        cSetEngine(cObject, engine.cObject);
    }

    private native long   cInit();
    private native void cSetEngine(long cObject, long cEngineObject);
    private native double cGetRetainRatio(long cParameter);
    private native void   cSetRetainRatio(long cParameter, double retainRatio);
    private native long   cGetSeed(long cParameter);
    private native void   cSetSeed(long cParameter, long seed);
}
/** @} */
