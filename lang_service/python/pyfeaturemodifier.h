/* file: pyfeaturemodifier.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef __PYFEATUREMODIFIER_H__
#define __PYFEATUREMODIFIER_H__

#ifdef _WIN32
#define NOMINMAX
#endif

#include <daal.h>
#include <algorithm>

namespace daal {
namespace data_management {
namespace modifiers {
namespace csv {
namespace interface1 {

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__MODIFIERS__CSV__FEATUREMODIFIER"></a>
 * \brief Base class for feature modifier, intended for inheritance from the user side
 */
class PyFeatureModifier : public FeatureModifier
{
public:
    /**
     * Create a new Modifier.
     * \param noof Number of output features,
     *             if > 0 it takes precedence over getNumberOfOutputFeatures
     * \param nor Number of rows to get processed in a chunk (in apply).
     *            Initially calls to apply might get smaller chunks,
     *            but they will never be greater than nor.
     **/
    PyFeatureModifier(size_t noof=0, size_t nor=1) :
        _noof(noof),
        _nor(std::max((size_t)1, nor)),
        _curr(0),
        _obuffers(_nor),
        _ibuffer(NULL)
    {
    }

    virtual ~PyFeatureModifier()
    {
        Py_XDECREF(_ibuffer);
    }

    /**
     * Overwrite this to provide the number of output features
     * (identical or each row). Gets overwritten by value provided
     * by constructor if that's > 0.
     * \return The number of input features
     */
    virtual size_t getNumberOfOutputFeatures() { return _noof; }

    /**
     * Applies the modifier.
     * Overwrite this with your modification logic. This method
     * gets called for a chunk input rows and might be called more than
     * once. It is expected to return as many results as it was provided with rows.
     * The number of values in each result must be identical for all results and
     * can be provided in the constructor or by overwriting getNumberOfOutputFeatures.
     * \param tokens A python list of tuples, each tuple represents one input row.
     *               The number of tuples is set/limited by the constructor.
     *               Each element in the tuple is a string value as read from data source.
     *               (like  [("red", "1"),("blue", "2")]
     * \return A Python sequence of sequences of floats
     *         like [2.2,3.3]
     */
    virtual PyObject * apply(PyObject * tokens) = 0;

protected:
    virtual void initialize(Config &config) DAAL_C11_OVERRIDE
    {
        if (_noof == 0 ) _noof = config.getNumberOfInputFeatures();
        config.setNumberOfOutputFeatures(getNumberOfOutputFeatures());
    }

    // Take the input args and call the python object.
    // Take the the python ouitput and copy result to the stored DAAL output buffers
    void apply_and_copy(size_t nor)
    {
        auto res = apply(_ibuffer);

        // We parse the Python sequence explicitly. We could SWIG let do this but it would imply
        // extra copying. Easier than to write a complicated typemap (if possible at all).

        // we need the GIL, we manipulate Python objects!
        auto gilstate = PyGILState_Ensure();
        if(PySequence_Check(res) && ! PyString_Check(res) && PySequence_Size(res) == nor) {
            for(size_t i = 0; i < nor; ++i) {
                daal::services::BufferView<DAAL_DATA_TYPE> outputBuffer = _obuffers[i];
                PyObject *row = PySequence_GetItem(res, i);
                if(PySequence_Check(row) && ! PyString_Check(row) && PySequence_Size(row) == _noof) {
                    for(size_t j = 0; j < _noof; ++j) {
                        PyObject * tmp = PySequence_GetItem(row, j);
                        outputBuffer[j] = PyFloat_AsDouble(tmp);
                        // We got a new reference -> release it
                        Py_DECREF(tmp);
                    }
                    Py_DECREF(row);
                } else {
                    PyGILState_Release(gilstate);
                    throw std::runtime_error("apply returned sequence of objects with wrong type or size");
                }
            }
        } else {
            PyGILState_Release(gilstate);
            throw std::runtime_error("apply returned object of wrong type or size");
        }
        // We are done, the Python result object can be released!
        Py_DECREF(res);
        PyGILState_Release(gilstate);
    }

    // generic C++'ish apply does collects a number of rows and then calls
    // the callback in Python witht the list of rows
    // We first create the Python Objects for input to the python callback
    // we then store the output buffer for this row
    // Eventually we call apply_and_copy to process all collected rows
    virtual void apply(Context &context) DAAL_C11_OVERRIDE
    {
        size_t n = context.getNumberOfTokens();

        // we create PyObjects explicitly, we need the GIL!
        auto gilstate = PyGILState_Ensure();

        if(_curr == 0) {
            Py_XDECREF(_ibuffer);
            _ibuffer = PyList_New(_nor);
        }

        // We explicitly create PyObjects here, simpler and probably faster than SWIG typemaps
        PyObject *tuple = PyTuple_New(n);
        for (size_t i = 0; i < n; i++) {
            PyTuple_SetItem(tuple, i, PyString_FromString(context.getToken(i).c_str()));
        }
        PyList_SetItem(_ibuffer, _curr, tuple);

        // done with Python stuff
        PyGILState_Release(gilstate);

        // we also need to store the output buffer for this line, we don't have access to it later otherwise
        _obuffers[_curr] = context.getOutputBuffer();

        // trigger processing if collected enough rows
        if(++_curr == _nor) {
            apply_and_copy(_nor);
            _curr = 0;
        }
    }

    // this is called multiple times after a varying number of calls to apply
    // because we do not know if this is the last time, we need to always process
    // what we have collected so far, even if < _nor
    virtual void finalize(Config &config) DAAL_C11_OVERRIDE
    {
        if(_curr > 0) {
            // if we have less rows than we wanted, we need to shrink our input list
            // ideally this does not happen often
            if(_curr < _nor) {
                auto gilstate = PyGILState_Ensure();
                PyObject * tmp = _ibuffer;
                _ibuffer = PyList_GetSlice(tmp, 0, _curr);
                Py_XDECREF(tmp);
                PyGILState_Release(gilstate);
            }
            apply_and_copy(_curr);
            _curr = 0;
        }
    }

private:
    size_t _noof, _nor, _curr; // # output features, # output rows, current row (< _nor)
    PyObject * _ibuffer;       // collecting our input rows/tuples in a PyList
    std::vector< daal::services::BufferView<DAAL_DATA_TYPE> > _obuffers; // DAAL's output buffers for each row one
};

} // namespace daal
} // namespace data_management
} // namespace modifiers
} // namespace csv
} // namespace interface1

#endif // __PYFEATUREMODIFIER_H__
