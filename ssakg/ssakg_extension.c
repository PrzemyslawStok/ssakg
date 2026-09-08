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

static void print_array(const PyArrayObject *array) {
    PyObject *obj_repr = PyObject_Repr((PyObject *) array);
    PyObject *str = PyUnicode_AsEncodedString(obj_repr, "utf-8", "~E~");
    printf("%s\n", PyBytes_AsString(str));

    Py_XDECREF(obj_repr);
    Py_XDECREF(str);
}

static void print_array_stride(const PyArrayObject *array, const npy_intp stride) {
    const npy_intp length = PyArray_SIZE(array);
    const npy_intp dims[1] = {length / stride};

    switch (stride) {
        case 2: {
            const PyArrayObject *array_to_print = (PyArrayObject *) PyArray_SimpleNewFromData(
                1, dims, NPY_INT16, PyArray_DATA(array));
            print_array(array_to_print);
            Py_DECREF(array_to_print);
            break;
        }
        case 4: {
            const PyArrayObject *array_to_print = (PyArrayObject *) PyArray_SimpleNewFromData(
                1, dims, NPY_INT32, PyArray_DATA(array));
            print_array(array_to_print);
            Py_DECREF(array_to_print);
            break;
        }
        case 8: {
            const PyArrayObject *array_to_print = (PyArrayObject *) PyArray_SimpleNewFromData(
                1, dims, NPY_INT64, PyArray_DATA(array));
            print_array(array_to_print);
            Py_DECREF(array_to_print);
            break;
        }
        default:
            break;
    }
}

static PyArrayObject *get_non_zeros_indices(const char *array, const npy_intp stride,
                                            const npy_intp graph_rows_no) {
    npy_intp non_zeros_no = 0;

    for (npy_intp i = 0; i < graph_rows_no; i++)
        switch (stride) {
            case 2: {
                if (*(npy_uint16 *) (array + i * stride) != 0) non_zeros_no++;
                break;
            }
            case 4: {
                if (*(npy_uint32 *) (array + i * stride) != 0) non_zeros_no++;
                break;
            }
            case 8: {
                if (*(npy_uint64 *) (array + i * stride) != 0) non_zeros_no++;
                break;
            }
            default:
                break;
        }


    // if (non_zeros_no == 0)
    //     return NULL;

    const npy_intp dims_non_zeros[] = {non_zeros_no};

    PyArray_Descr *dtype = PyArray_DescrNewFromType(NPY_INTP);
    PyArrayObject *non_zeros_object = (PyArrayObject *) PyArray_NewFromDescr(
        &PyArray_Type, dtype, 1, dims_non_zeros,NULL,NULL,NPY_DEFAULT,NULL);

    npy_intp *non_zeros_array = PyArray_DATA(non_zeros_object);

    if (non_zeros_array == NULL)
        return NULL;

    npy_intp j = 0;

    for (npy_intp i = 0; i < graph_rows_no; i++) {
        switch (stride) {
            case 2: {
                if (*(npy_uint16 *) (array + i * stride) != 0) {
                    non_zeros_array[j] = i;
                    j++;
                }
                break;
            }
            case 4: {
                if (*(npy_uint32 *) (array + i * stride) != 0) {
                    non_zeros_array[j] = i;
                    j++;
                }
                break;
            }
            case 8: {
                if (*(npy_uint64 *) (array + i * stride) != 0) {
                    non_zeros_array[j] = i;
                    j++;
                }
                break;
            }
            default:
                break;
        }
    }

    return non_zeros_object;
}

static PyArrayObject *create_prod_array_object_triangle(const npy_uint16 *graph_array, const npy_intp graph_rows_no,
                                                        const npy_intp graph_cols_no, const npy_uint32 *context_array,
                                                        const npy_intp context_length) {
    PyArray_Descr *dtype = PyArray_DescrNewFromType(NPY_UINT16);

    const npy_intp dims[] = {graph_rows_no};
    PyArrayObject *prod_object = (PyArrayObject *) PyArray_NewFromDescr(&PyArray_Type, dtype, 1, dims,NULL,NULL,
                                                                        NPY_DEFAULT,NULL);

    npy_uint16 *prod_array = PyArray_DATA(prod_object);

    for (npy_intp i = 0; i < graph_rows_no; i++)
        prod_array[i] = 1;

    for (npy_intp i = 0; i < graph_rows_no; i++) {
        for (npy_intp j = 0; j < context_length; j++) {
            const npy_intp index = i * graph_cols_no + context_array[j];
            const npy_intp index_translated = context_array[j] * graph_cols_no + i;
            npy_uint32 element = graph_array[index] + graph_array[index_translated];

            if (i == context_array[j]) element = 1;

            prod_array[i] *= element != 0;
            if (prod_array[i] == 0) break;
        }
    }

    return prod_object;
}

static PyObject *get_unsorted_elements(PyObject *self, PyObject *args) {
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

    const npy_intp stride = PyArray_STRIDE(graph, 1);

    const npy_intp graph_rows_no = PyArray_SHAPE(graph)[0];
    const npy_intp graph_cols_no = PyArray_SHAPE(graph)[1];
    const npy_intp context_length = PyArray_SHAPE(context)[0];

    const npy_uint32 *context_array = (npy_uint32 *) PyArray_DATA(context);
    const npy_uint16 *graph_array = (npy_uint16 *) PyArray_DATA(graph);

    const PyArrayObject *prod_object = create_prod_array_object_triangle(graph_array, graph_rows_no, graph_cols_no,
                                                                         context_array, context_length);

    const char *prod_array = PyArray_DATA(prod_object);
    const PyArrayObject *non_zeros_object = get_non_zeros_indices(prod_array, stride, graph_rows_no);

    Py_DECREF(prod_object);

    return (PyObject *) non_zeros_object;
}

static void print_array_1d(const PyArrayObject *array) {
    const npy_intp length = PyArray_SIZE(array);
    const npy_intp stride = PyArray_STRIDE(array, 0);
    char *array_data = PyArray_DATA(array);

    printf("[");
    for (npy_intp i = 0; i < length; i++) {
        switch (PyArray_TYPE(array)) {
            case NPY_UINT16: {
                const npy_uint16 element = *(npy_uint16 *) (array_data + i * stride);
                printf("%d", element);
                break;
            }
            case NPY_UINT32: {
                const npy_uint32 element = *(npy_uint32 *) (array_data + i * stride);
                printf("%d", element);
                break;
            }
            case NPY_UINT64:
            case NPY_INTP: {
                const npy_uint64 element = *(npy_uint64 *) (array_data + i * stride);
                printf("%lu", element);
                break;
            }
            default:
                printf("Incorrect data type: graph datatype should be \"uint16\", \"uint32\", \"uint64\"");
                return;
        }

        if (i < length - 1)
            printf(" ");
        else
            printf("]");
    }
    printf("\n");
}

static PyObject *find_correct_bit_base(const PyArrayObject *array_object, const npy_intp stride, npy_intp min_range,
                                       npy_intp max_range) {
    const npy_intp no_bits = 8 * stride;
    const char *row_array = PyArray_DATA(array_object);


    const npy_intp row_len = PyArray_SHAPE(array_object)[0];

    if (min_range > row_len)
        min_range = 0;
    if (max_range > row_len)
        max_range = row_len;

    const npy_intp dims[] = {no_bits};

    PyArrayObject *bits_no_object = (PyArrayObject *) PyArray_ZEROS(1, dims, NPY_UINT16, 0);
    npy_uint16 *bits_no = PyArray_DATA(bits_no_object);

    PyArrayObject *bits_mask_object = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_UINT64);
    npy_uint64 *bits_mask = PyArray_DATA(bits_mask_object);

    for (npy_uint64 i = 0; i < no_bits; i++)
        bits_mask[i] = (npy_uint64) 1 << i;

    for (npy_intp j = min_range; j < max_range; j++) {
        switch (stride) {
            case 2: {
                const npy_uint16 element = *(npy_uint16 *) (row_array + j * stride);

                for (npy_intp i = 0; i < no_bits; i++)
                    if ((element & bits_mask[i]) > 0) bits_no[i]++;
                break;
            }
            case 4: {
                const npy_uint32 element = *(npy_uint32 *) (row_array + j * stride);
                for (npy_intp i = 0; i < no_bits; i++)
                    if ((element & bits_mask[i]) > 0) bits_no[i]++;
                break;
            }
            case 8: {
                const npy_uint64 element = *(npy_uint64 *) (row_array + j * stride);

                for (npy_intp i = 0; i < no_bits; i++)
                    if ((element & bits_mask[i]) > 0) bits_no[i]++;
                break;
            }
            default:
                break;
        }
    }

    Py_DECREF(bits_mask_object);

    PyObject *bit_max = PyArray_ArgMax(bits_no_object, 0, NULL);

    Py_DECREF(bits_no_object);
    // printf("bit_max: %lu\n", PyLong_AsLongLong(bit_max));

    return bit_max;
}

static PyObject *findCorrectBit(PyObject *self, PyObject *args) {
    PyArrayObject *row;

    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &row)) {
        PyErr_SetString(PyExc_TypeError,
                        "This function requires numpy array as a parameters.");
        return NULL;
    }
    if (!(PyArray_TYPE(row) == NPY_UINT16 || PyArray_TYPE(row) == NPY_UINT32 || PyArray_TYPE(row) ==
          NPY_UINT64)) {
        PyErr_SetString(PyExc_TypeError,
                        "Incorrect data type: graph datatype should be \"uint16|uint32|uint64)\"");
        return NULL;
    }

    const npy_intp stride = PyArray_STRIDE(row, 0);

    PyObject *bit_max = find_correct_bit_base(row, stride, 0, PyArray_SHAPE(row)[0]);
    return bit_max;
}

static PyArrayObject *create_prod_array_object_square(const char *graph_array, const npy_intp stride,
                                                      const npy_intp graph_rows_no,
                                                      const npy_intp graph_cols_no, const npy_intp *context_array,
                                                      const npy_intp context_length) {
    PyArray_Descr *dtype = PyArray_DescrNewFromType(NPY_UINT16);

    const npy_intp dims[] = {graph_rows_no};
    PyArrayObject *prod_object = (PyArrayObject *) PyArray_NewFromDescr(&PyArray_Type, dtype, 1, dims,NULL,NULL,
                                                                        NPY_DEFAULT,NULL);
    if (prod_object == NULL) {
        Py_DECREF(dtype);
        PyErr_SetString(PyExc_MemoryError, "Unable to allocate memory for array.");
        return NULL;
    }

    npy_uint16 *prod_array = PyArray_DATA(prod_object);

    for (npy_intp i = 0; i < graph_rows_no; i++)
        prod_array[i] = 1;

    for (npy_intp i = 0; i < graph_rows_no; i++) {
        for (npy_intp j = 0; j < context_length; j++) {
            const npy_intp index_translated = context_array[j] * graph_cols_no + i;
            switch (stride) {
                case 2: {
                    const npy_uint16 element = *(npy_uint16 *) (graph_array + index_translated * stride);
                    prod_array[i] *= element != 0;
                    break;
                }
                case 4: {
                    const npy_uint32 element = *(npy_uint32 *) (graph_array + index_translated * stride);
                    prod_array[i] *= element != 0;
                    break;
                }
                case 8: {
                    const npy_uint64 element = *(npy_uint64 *) (graph_array + index_translated * stride);
                    prod_array[i] *= element != 0;
                    break;
                }
                default:
                    break;
            }
        }
    }

    return prod_object;
}

static PyArrayObject *create_limited_graph(const char *graph_array, const npy_intp stride, const npy_intp graph_cols_no,
                                           const npy_intp *context_array,
                                           const npy_intp context_length, const npy_intp *non_zeros_array,
                                           const npy_intp non_zeros_no) {
    PyArray_Descr *dtype = PyArray_DescrNewFromType(NPY_UINT8);
    const npy_intp dims_limited_graph[] = {non_zeros_no * context_length * stride};

    PyArrayObject *limited_graph_object = (PyArrayObject *) PyArray_NewFromDescr(
        &PyArray_Type, dtype, 1, dims_limited_graph,NULL,NULL,NPY_DEFAULT,NULL);

    char *limited_graph = PyArray_DATA(limited_graph_object);

    for (npy_intp j = 0; j < context_length; j++)
        for (npy_intp i = 0; i < non_zeros_no; i++) {
            const npy_intp limited_index = j * non_zeros_no + i;
            const npy_intp graph_index = context_array[j] * graph_cols_no + non_zeros_array[i];
            switch (stride) {
                case 2: {
                    *(npy_uint16 *) (limited_graph + limited_index * stride) = *(npy_uint16 *) (
                        graph_array + graph_index * stride);
                    break;
                }
                case 4: {
                    *(npy_uint32 *) (limited_graph + limited_index * stride) = *(npy_uint32 *) (
                        graph_array + graph_index * stride);
                    break;
                }
                case 8: {
                    *(npy_uint64 *) (limited_graph + limited_index * stride) = *(npy_uint64 *) (
                        graph_array + graph_index * stride);
                    break;
                }
                default:
                    break;
            }
        }
    return limited_graph_object;
}

static void print_array_2d(const npy_uint16 *array, const npy_intp rows, const npy_intp cols) {
    printf("[");
    for (npy_intp j = 0; j < cols; j++) {
        if (j > 0)printf(" ");
        printf("[");
        for (npy_intp i = 0; i < rows; i++) {
            const npy_intp index = j * rows + i;
            if (i < rows - 1)
                printf("%d ", array[index]);
            else
                printf("%d]", array[index]);
        }
        if (j < cols - 1)
            printf("\n");
    }
    printf("]\n");
}

static PyArrayObject *filtered_array(PyArrayObject *array_object, const npy_intp stride, const npy_intp rows,
                                     const npy_intp cols) {
    PyArrayObject *filtered_object = (PyArrayObject *) PyArray_NewCopy(array_object, NPY_CORDER);
    char *array_data = PyArray_DATA(filtered_object);

    for (npy_intp j = 0; j < rows; j++) {
        const npy_intp min_range = j * cols;
        const npy_intp max_range = j * cols + cols;

        PyObject *bit_max = find_correct_bit_base(filtered_object, stride, min_range, max_range);

        const npy_uint16 bit_max_value = PyLong_AsLong(bit_max);

        for (npy_intp i = 0; i < rows * cols; i++) {
            switch (stride) {
                case 2: {
                    if (i >= min_range && i < max_range) {
                        npy_uint16 *element = (npy_uint16 *) (array_data + i * stride);
                        *element = *element & 1 << bit_max_value;
                    } else if (i >= max_range) {
                        npy_uint16 *element = (npy_uint16 *) (array_data + i * stride);
                        *element = *element & ~(1 << bit_max_value);
                    }
                    break;
                }
                case 4: {
                    if (i >= min_range && i < max_range) {
                        npy_uint32 *element = (npy_uint32 *) (array_data + i * stride);
                        *element = *element & 1 << bit_max_value;
                    } else if (i >= max_range) {
                        npy_uint32 *element = (npy_uint32 *) (array_data + i * stride);
                        *element = *element & ~(1 << bit_max_value);
                    }
                    break;
                }
                case 8: {
                    if (i >= min_range && i < max_range) {
                        npy_uint64 *element = (npy_uint64 *) (array_data + i * stride);
                        *element = *element & (npy_uint64) 1 << (npy_uint64) bit_max_value;
                    } else if (i >= max_range) {
                        npy_uint64 *element = (npy_uint64 *) (array_data + i * stride);
                        *element = *element & ~((npy_uint64) 1 << (npy_uint64) bit_max_value);
                    }
                    break;
                }
                default:
                    break;
            }
        }
    }
    return filtered_object;
}

static PyArrayObject *create_prod_array(const char *array, const npy_intp stride, const npy_intp rows,
                                        const npy_intp cols) {
    PyArray_Descr *dtype = PyArray_DescrNewFromType(NPY_UINT16);
    const npy_intp dims[] = {cols};

    PyArrayObject *prod_object = (PyArrayObject *) PyArray_NewFromDescr(&PyArray_Type, dtype, 1, dims,NULL,NULL,
                                                                        NPY_DEFAULT,NULL);
    npy_uint16 *prod_array = PyArray_DATA(prod_object);

    for (npy_intp i = 0; i < cols; i++)
        prod_array[i] = 1;

    for (npy_intp j = 0; j < rows; j++) {
        for (npy_intp i = 0; i < cols; i++) {
            const npy_intp index = (j * cols + i) * stride;
            switch (stride) {
                case 2: {
                    prod_array[i] *= *(npy_uint16 *) (array + index) != 0;
                    break;
                }
                case 4: {
                    prod_array[i] *= *(npy_uint32 *) (array + index) != 0;
                    break;
                }
                case 8: {
                    prod_array[i] *= *(npy_uint64 *) (array + index) != 0;
                    break;
                }
                default:
                    break;
            }
        }
    }

    return prod_object;
}

static PyArrayObject *array_from_indices(const npy_intp *array, const npy_intp *indices,
                                         const npy_intp indices_length) {
    PyArray_Descr *dtype = PyArray_DescrNewFromType(NPY_INTP);
    const npy_intp dims[] = {indices_length};

    PyArrayObject *indices_array = (PyArrayObject *) PyArray_NewFromDescr(&PyArray_Type, dtype, 1, dims,NULL,NULL,
                                                                          NPY_DEFAULT,NULL);
    npy_intp *indices_array_data = PyArray_DATA(indices_array);

    for (npy_intp i = 0; i < indices_length; i++)
        indices_array_data[i] = array[indices[i]];

    return indices_array;
}

static PyArrayObject *
get_subgraph_from_sequence(const char *graph_array, const npy_intp stride, const npy_intp graph_rows,
                           const npy_intp *sequence,
                           const npy_intp length) {
    PyArray_Descr *dtype = PyArray_DescrNewFromType(NPY_UINT8);
    const npy_intp dims[] = {length * length * stride};

    PyArrayObject *subgraph_array = (PyArrayObject *) PyArray_NewFromDescr(&PyArray_Type, dtype, 1, dims,NULL,NULL,
                                                                           NPY_DEFAULT,NULL);
    char *subgraph_array_data = PyArray_DATA(subgraph_array);

    for (npy_intp i = 0; i < length; i++) {
        for (npy_intp j = 0; j < length; j++) {
            const npy_intp index = i * length + j;
            const npy_intp graph_index = sequence[i] * graph_rows + sequence[j];

            switch (stride) {
                case 2: {
                    *(npy_uint16 *) (subgraph_array_data + index * stride) = *(npy_uint16 *) (
                        graph_array + graph_index * stride);
                    break;
                }
                case 4: {
                    *(npy_uint32 *) (subgraph_array_data + index * stride) = *(npy_uint32 *) (
                        graph_array + graph_index * stride);
                    break;
                }
                case 8: {
                    *(npy_uint64 *) (subgraph_array_data + index * stride) = *(npy_uint64 *) (
                        graph_array + graph_index * stride);
                    break;
                }
                default:
                    break;
            }
        }
    }

    return subgraph_array;
}

static PyArrayObject *get_sorted_indices_from_first_column(const char *array, const npy_intp stride,
                                                           const npy_intp rows,
                                                           const npy_intp cols) {
    PyArray_Descr *dtype = PyArray_DescrNewFromType(NPY_UINT64);
    const npy_intp dims[] = {rows};

    PyArrayObject *first_column = (PyArrayObject *) PyArray_NewFromDescr(&PyArray_Type, dtype, 1, dims,NULL,NULL,
                                                                         NPY_DEFAULT,NULL);
    npy_uint64 *first_column_data = PyArray_DATA(first_column);

    for (npy_intp i = 0; i < rows; i++) {
        const npy_intp index = i * cols;
        switch (stride) {
            case 2: {
                first_column_data[i] = (npy_uint64) *(npy_uint16 *) (array + index * stride);
                break;
            }
            case 4: {
                first_column_data[i] = *(npy_uint32 *) (array + index * stride);
                break;
            }
            case 8: {
                first_column_data[i] = *(npy_uint64 *) (array + index * stride);
                break;
            }
            default:
                break;
        }
    }

    PyArrayObject *indices = (PyArrayObject *) PyArray_ArgSort(first_column, 0, NPY_QUICKSORT);
    Py_DECREF(first_column);
    return indices;
}

static PyObject *getSortedElementsBits(PyObject *self, PyObject *args) {
    PyArrayObject *graph, *context;
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &graph,
                          &PyArray_Type, &context)) {
        PyErr_SetString(PyExc_TypeError,
                        "This function requires two numpy arrays as a parameters.");
        return NULL;
    }

    if (!(PyArray_TYPE(graph) == NPY_UINT16 || PyArray_TYPE(graph) == NPY_UINT32 || PyArray_TYPE(graph) ==
          NPY_UINT64)) {
        PyErr_SetString(PyExc_TypeError,
                        "Incorrect data type: graph datatype should be \"uint16|uint32|uint64)\"");
        return NULL;
    }

    if (PyArray_TYPE(context) != NPY_UINTP) {
        PyErr_SetString(PyExc_TypeError,
                        "Incorrect data type: context datatype should be \"uintp\"");
        return NULL;
    }

    //Stride of main element graph
    const npy_intp stride = PyArray_STRIDE(graph, 1);

    const npy_intp graph_rows_no = PyArray_SHAPE(graph)[0];
    const npy_intp graph_cols_no = PyArray_SHAPE(graph)[1];
    const npy_intp context_length = PyArray_SHAPE(context)[0];

    // const PyArrayObject *context_1 = (PyArrayObject *) PyArray_CastToType((PyArrayObject *) context,
    //                                                          PyArray_DescrNewFromType(NPY_UINT32), 0);

    const npy_intp *context_array = PyArray_DATA(context);
    const char *graph_array = PyArray_DATA(graph);

    const PyArrayObject *prod_object = create_prod_array_object_square(graph_array, stride, graph_rows_no,
                                                                       graph_cols_no,
                                                                       context_array, context_length);
    const char *prod_array = PyArray_DATA(prod_object);
    // prod object is always type npy_uint16, stride is equal 2
    const PyArrayObject *non_zeros_object = get_non_zeros_indices(prod_array, 2, graph_rows_no);
    const npy_intp non_zeros_no = PyArray_SHAPE(non_zeros_object)[0];

    const npy_intp *non_zeros_array = PyArray_DATA(non_zeros_object);

    PyArrayObject *limited_graph_object = create_limited_graph(graph_array, stride, graph_cols_no, context_array,
                                                               context_length, non_zeros_array, non_zeros_no);
    const PyArrayObject *filtered_limited_graph = filtered_array(limited_graph_object, stride, context_length,
                                                                 non_zeros_no);

    const PyArrayObject *filtered_prod = create_prod_array(PyArray_DATA(filtered_limited_graph), stride, context_length,
                                                           non_zeros_no);

    // prod object is always type npy_uint16, stride is equal 2
    const PyArrayObject *filtered_non_zeros = get_non_zeros_indices(PyArray_DATA(filtered_prod), 2, non_zeros_no);
    const npy_intp filtered_non_zeros_no = PyArray_SHAPE(filtered_non_zeros)[0];
    const PyArrayObject *filtered_indices = array_from_indices(non_zeros_array, PyArray_DATA(filtered_non_zeros),
                                                               filtered_non_zeros_no);

    PyArrayObject *sequence_matrix = get_subgraph_from_sequence(graph_array, stride, graph_cols_no,
                                                                PyArray_DATA(filtered_indices),
                                                                filtered_non_zeros_no);

    const PyArrayObject *filtered_sequence_matrix = filtered_array(sequence_matrix, stride, filtered_non_zeros_no,
                                                                   filtered_non_zeros_no);

    // print_array_stride(filtered_sequence_matrix, stride);

    const PyArrayObject *sorted_indices = get_sorted_indices_from_first_column(
        PyArray_DATA(filtered_sequence_matrix), stride, filtered_non_zeros_no, filtered_non_zeros_no);

    const PyArrayObject *sorted_sequence = array_from_indices(PyArray_DATA(filtered_indices),
                                                              PyArray_DATA(sorted_indices), filtered_non_zeros_no);
    // print_array(sorted_sequence);
    // print_array_1d(sorted_sequence);

    Py_DECREF(non_zeros_object);
    Py_DECREF(sorted_indices);
    Py_DECREF(filtered_limited_graph);
    Py_DECREF(limited_graph_object);
    Py_DECREF(prod_object);
    Py_DECREF(sequence_matrix);
    Py_DECREF(filtered_sequence_matrix);

    PyObject *result = PyTuple_New(2);

    PyTuple_SetItem(result, 0, (PyObject *) sorted_sequence);
    PyTuple_SetItem(result, 1, (PyObject *) filtered_indices);

    return result;
}

static PyMethodDef SSAKG_methods[] = {
    {"get_unsorted_elements", get_unsorted_elements, METH_VARARGS, "SSAKG C get_unsorted_elements function"},
    {"find_correct_bit", findCorrectBit, METH_VARARGS, "SSAKG C find_correct_bits function"},
    {
        "get_sorted_elements_bits", getSortedElementsBits, METH_VARARGS,
        "SSAKG C get_unsorted_elements function bits version"
    },
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
