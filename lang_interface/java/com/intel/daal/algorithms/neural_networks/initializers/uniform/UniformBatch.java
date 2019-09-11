/* file: UniformBatch.java */
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
 * @defgroup initializers_uniform_batch Batch
 * @ingroup initializers_uniform
 * @{
 */
/**
 * @brief Contains classes for the uniform initializer
 */
package com.intel.daal.algorithms.neural_networks.initializers.uniform;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.neural_networks.initializers.Input;
import com.intel.daal.algorithms.neural_networks.initializers.Result;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__INITIALIZERS__UNIFORM__UNIFORMBATCH"></a>
 * \brief Provides methods for uniform initializer computations in the batch processing mode
 *
 * \par References
 *      - @ref com.intel.daal.algorithms.neural_networks.initializers.Input class
 */
public class UniformBatch extends com.intel.daal.algorithms.neural_networks.initializers.InitializerIface {
    public  UniformParameter    parameter; /*!< UniformParameters of the uniform initializer */
    public  UniformMethod       method;    /*!< Computation method for the initializer */
    private Precision    prec;      /*!< Data type to use in intermediate computations for the initializer */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs uniform initializer by copying input objects and parameters of another uniform initializer
     * @param context Context to manage the uniform initializer
     * @param other   An initializer to be used as the source to initialize the input objects
     *                and parameters of this initializer
     */
    public UniformBatch(DaalContext context, UniformBatch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input = new Input(context, cGetInput(cObject));
        parameter = new UniformParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
    * Constructs the uniform initializer
    * @param context    Context to manage the initializer
    * @param cls        Data type to use in intermediate computations for the initializer, Double.class or Float.class
    * @param method     The initializer computation method, @ref UniformMethod
    * @param a          Left bound of the interval
    * @param b          Right bound of the interval
    */
    public UniformBatch(DaalContext context, Class<? extends Number> cls, UniformMethod method, double a, double b) {
        super(context);

        constructBatch(context, cls, method, a, b);
    }

    private void constructBatch(DaalContext context, Class<? extends Number> cls, UniformMethod method, double a, double b) {
        this.method = method;

        if (method != UniformMethod.defaultDense) {
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

        this.cObject = cInit(prec.getValue(), method.getValue(), a, b);
        input = new Input(context, cGetInput(cObject));
        parameter = new UniformParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
        parameter.setA(a);
        parameter.setB(b);
    }

    /**
     * Computes the result of the uniform initializer
     * @return  Uniform initializer result
     */
    @Override
    public Result compute() {
        super.compute();
        return new Result(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Returns the newly allocated uniform initializer
     * with a copy of input objects and parameters of this uniform initializer
     * @param context    Context to manage the initializer
     * @return The newly allocated uniform initializer
     */
    @Override
    public UniformBatch clone(DaalContext context) {
        return new UniformBatch(context, this);
    }

    private native long cInit(int prec, int method, double a, double b);
    private native long cInitParameter(long cAlgorithm, int prec, int method);
    private native long cGetResult(long cAlgorithm, int prec, int method);
    private native long cClone(long algAddr, int prec, int method);
}
/** @} */
