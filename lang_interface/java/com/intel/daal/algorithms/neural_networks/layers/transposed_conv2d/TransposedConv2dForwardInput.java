/* file: TransposedConv2dForwardInput.java */
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
 * @defgroup transposed_conv2d_forward Forward Two-dimensional Transposed Convolution Layer
 * @brief Contains classes for the forward 2D transposed convolution layer
 * @ingroup transposed_conv2d
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.transposed_conv2d;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__TRANSPOSED_CONV2D__TRANSPOSEDCONV2DFORWARDINPUT"></a>
 * @brief %Input object for the forward 2D transposed convolution layer
 */
public class TransposedConv2dForwardInput extends com.intel.daal.algorithms.neural_networks.layers.ForwardInput {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the forward 2D transposed convolution layer input
     * @param context   Context to manage the forward 2D transposed convolution layer input
     * @param cObject   Address of C++ forward input
     */
    public TransposedConv2dForwardInput(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Returns dimensions of weights tensor
     * @param parameter  Layer parameter
     * @return Dimensions of weights tensor
     */
    public long[] getWeightsSizes(TransposedConv2dParameter parameter)
    {
        return cGetWeightsSizes(cObject, parameter.getCObject());
    }

    /**
     * Returns dimensions of biases tensor
     * @param parameter  Layer parameter
     * @return Dimensions of biases tensor
     */
    public long[] getBiasesSizes(TransposedConv2dParameter parameter)
    {
        return cGetBiasesSizes(cObject, parameter.getCObject());
    }

    private native long[] cGetWeightsSizes(long cObject, long cParameter);
    private native long[] cGetBiasesSizes(long cObject, long cParameter);
}
/** @} */
