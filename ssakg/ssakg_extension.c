// Copyright 2024 The SSAKG Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <numpy/arrayobject.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

static PyObject *getUnsortedElements(PyObject* self, PyObject* args){
    PyArrayObject *graph, *context;
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &graph,
                          &PyArray_Type, &context)) {
        PyErr_SetString(PyExc_TypeError,
                        "This function requires two numpy arrays as a parameters.");
        return NULL;
    }

    if (PyArray_TYPE(graph) != NPY_UINT16) {
        PyErr_SetString(PyExc_TypeError,
                        "Incorrect data type: graph datatype should be \"uint16\"");
        return NULL;
    }

    if (PyArray_TYPE(context) != NPY_UINT32) {
        PyErr_SetString(PyExc_TypeError,
                        "Incorrect data type: context datatype should be \"uint32\"");
        return NULL;
    }

    PyArray_Descr* dtype = PyArray_DescrNewFromType(NPY_UINT16);

    npy_uint64 graph_rows_no = PyArray_SHAPE(graph)[0];
    npy_uint64 graph_cols_no = PyArray_SHAPE(graph)[1];
    npy_uint64 context_length = PyArray_SHAPE(context)[0];

    npy_intp dims[] = {graph_rows_no};
    PyArrayObject * prod_object = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type, dtype,1,dims,NULL,NULL,NPY_DEFAULT,NULL);

    npy_uint16 * prod_array = (npy_uint16*)PyArray_DATA(prod_object);
    npy_uint32 * context_array = (npy_uint32*)PyArray_DATA(context);
    npy_uint16 * graph_array = (npy_uint16*)PyArray_DATA(graph);

    for(npy_uint64 i=0;i<graph_rows_no;i++)
        prod_array[i] = 1;

     for(npy_uint64 i=0;i<graph_rows_no;i++){
        for(npy_uint64 j=0;j<context_length;j++){
            npy_uint64 index = i*graph_cols_no+context_array[j];
            npy_uint64 index_translated = context_array[j]*graph_cols_no+i;
            npy_uint32 element = graph_array[index] + graph_array[index_translated];

            if(i == context_array[j]) element=1;

            prod_array[i] *= element != 0;
            if(prod_array[i] == 0) break;
        }
     }

     npy_uint64 non_zeros_no = 0;
     for(npy_uint64 i=0;i<graph_rows_no;i++)
        if(prod_array[i] != 0) non_zeros_no++;

     npy_intp dims_non_zeros[] = {non_zeros_no};

     PyArray_Descr* dtype32 = PyArray_DescrNewFromType(NPY_UINT32);
     PyArrayObject * non_zeros_object = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type, dtype32,1,dims_non_zeros,NULL,NULL,NPY_DEFAULT,NULL);
     npy_uint32 * non_zeros_array = (npy_uint32*)PyArray_DATA(non_zeros_object);

     npy_uint64 j = 0;

     for(npy_uint64 i=0;i<graph_rows_no;i++){
        if(prod_array[i] != 0){
            non_zeros_array[j]=i;
            j++;
        }
     }
     
    return (PyObject *)non_zeros_object;
}

static PyMethodDef SSAKG_methods[] = {
    {"get_unsorted_elements", getUnsortedElements, METH_VARARGS, "SSAKG C get_unsorted_elements function"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef ssakg_module = {
    PyModuleDef_HEAD_INIT,
    "ssakg extension numpy",
    "Python interface for the ssakg C extensions library",
    -1,
    SSAKG_methods
};

PyMODINIT_FUNC PyInit_ssakg_extension(void) {
    import_array();
    PyObject *module = PyModule_Create(&ssakg_module);
    return module;
}