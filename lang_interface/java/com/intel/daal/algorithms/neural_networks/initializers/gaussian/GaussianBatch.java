/* file: GaussianBatch.java */
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
 * @defgroup initializers_gaussian_batch Batch
 * @ingroup initializers_gaussian
 * @{
 */
/**
 * @brief Contains classes for the gaussian initializer
 */
package com.intel.daal.algorithms.neural_networks.initializers.gaussian;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.neural_networks.initializers.Input;
import com.intel.daal.algorithms.neural_networks.initializers.Result;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__INITIALIZERS__GAUSSIAN__GAUSSIANBATCH"></a>
 * \brief Provides methods for gaussian initializer computations in the batch processing mode
 *
 * \par References
 *      - @ref com.intel.daal.algorithms.neural_networks.initializers.Input class
 */
public class GaussianBatch extends com.intel.daal.algorithms.neural_networks.initializers.InitializerIface {
    public  GaussianParameter    parameter; /*!< GaussianParameters of the gaussian initializer */
    public  GaussianMethod       method;    /*!< Computation method for the initializer */
    private Precision    prec;      /*!< Data type to use in intermediate computations for the initializer */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs gaussian initializer by copying input objects and parameters of another gaussian initializer
     * @param context Context to manage the gaussian initializer
     * @param other   An initializer to be used as the source to initialize the input objects
     *                and parameters of this initializer
     */
    public GaussianBatch(DaalContext context, GaussianBatch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input = new Input(context, cGetInput(cObject));
        parameter = new GaussianParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the gaussian initializer
     * @param context    Context to manage the initializer
     * @param cls        Data type to use in intermediate computations for the initializer, Double.class or Float.class
     * @param method     The initializer computation method, @ref GaussianMethod
     * @param a          The distribution mean
     * @param sigma      The standard deviation of the distribution
     */
    public GaussianBatch(DaalContext context, Class<? extends Number> cls, GaussianMethod method, double a, double sigma) {
        super(context);
        constructBatch(context, cls, method, a, sigma);
    }

    private void constructBatch(DaalContext context, Class<? extends Number> cls, GaussianMethod method, double a, double sigma) {
        this.method = method;

        if (method != GaussianMethod.defaultDense) {
            throw new IllegalArgumentException("method unsupported");
        }
        if (cls != Double.class && cls != Float.class) {
            throw new IllegalArgumentException("type unsupported");
        }

        if (cls == Double.class) {
            prec = Precision.doublePrecision;
        }
        else {
            prec = Precision.singlePrecision;
        }

        this.cObject = cInit(prec.getValue(), method.getValue(), a, sigma);
        input = new Input(context, cGetInput(cObject));
        parameter = new GaussianParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
        parameter.setA(a);
        parameter.setSigma(sigma);
    }

    /**
     * Computes the result of the gaussian initializer
     * @return  Gaussian initializer result
     */
    @Override
    public Result compute() {
        super.compute();
        return new Result(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Returns the newly allocated gaussian initializer
     * with a copy of input objects and parameters of this gaussian initializer
     * @param context    Context to manage the initializer
     * @return The newly allocated gaussian initializer
     */
    @Override
    public GaussianBatch clone(DaalContext context) {
        return new GaussianBatch(context, this);
    }

    private native long cInit(int prec, int method, double a, double sigma);
    private native long cInitParameter(long cAlgorithm, int prec, int method);
    private native long cGetResult(long cAlgorithm, int prec, int method);
    private native long cClone(long algAddr, int prec, int method);
}
/** @} */
