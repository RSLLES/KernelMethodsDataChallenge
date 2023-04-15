#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <string>
#include <iostream>
#include "lib/tree/tree.hh"
#include "ted.hpp"
#include "tree_helper.hpp"
#include <numpy/arrayobject.h>
#include <omp.h>
#include <ctime>
#include <cmath>

using base_type = uint32_t;

static PyObject *computeVectorized(PyObject *self, PyObject *args)
{
    import_array();
    PyObject *list1;
    PyObject *list2 = NULL;

    if (!PyArg_ParseTuple(args, "O|O", &list1, &list2))
    {
        return NULL;
    }

    bool fit = false;
    if (list2 == NULL)
    {
        // If list2 is not provided, set it equal to list1
        Py_INCREF(list1);
        list2 = list1;
        fit = true;
    }

    // Convert input lists to C arrays for better performance
    int len1 = PyList_Size(list1);
    std::vector<std::vector<std::string>> arr1(len1);
    for (int i = 0; i < len1; ++i)
    {
        PyObject *sublist = PyList_GetItem(list1, i);
        int sublist_size = PyList_Size(sublist);
        for (int j = 0; j < sublist_size; ++j)
        {
            arr1[i].push_back(std::string(PyUnicode_AsUTF8(PyList_GetItem(sublist, j))));
        }
    }
    int len2 = PyList_Size(list2);
    std::vector<std::vector<std::string>> arr2(len2);
    for (int i = 0; i < len2; ++i)
    {
        PyObject *sublist = PyList_GetItem(list2, i);
        int sublist_size = PyList_Size(sublist);
        for (int j = 0; j < sublist_size; ++j)
        {
            arr2[i].push_back(std::string(PyUnicode_AsUTF8(PyList_GetItem(sublist, j))));
        }
    }

    std::vector<std::vector<tree<base_type>>> v1s(len1), v2s(len2);

    for (int i = 0; i < len1; ++i)
    {
        std::vector<tree<base_type>> v1(arr1[i].size());
        for (int k = 0; k < arr1[i].size(); ++k)
        {
            v1[k] = build_tree<base_type>(arr1[i][k]);
        }
        v1s[i] = std::move(v1);
    }

    for (int i = 0; i < len2; ++i)
    {
        std::vector<tree<base_type>> v2(arr2[i].size());
        for (int k = 0; k < arr2[i].size(); ++k)
        {
            v2[k] = build_tree<base_type>(arr2[i][k]);
        }
        v2s[i] = std::move(v2);
    }

    std::vector<std::vector<double>> res(len1, std::vector<double>(len2));
    int numThreads = omp_get_num_procs() - 2;
    int numIterations = fit ? len1 * (len2 + 1) / 2 : len1 * len2;
    int completedIterations = 0;
    double timePerIteration = 0.0;
    time_t startTime = time(nullptr);
    LookupTable lookup;

#pragma omp parallel for num_threads(numThreads) schedule(dynamic, 64)
    for (int i = 0; i < len1; ++i)
    {
        for (int j = fit ? i : 0; j < len2; ++j)
        {

            res[i][j] = kernel(v1s[i], v2s[j], lookup);
            if (fit)
            {
                res[j][i] = res[i][j];
            }

            bool print = false;
#pragma omp critical
            {
                ++completedIterations;
                print = (completedIterations % 500 == 0);
            }

            // Update progress bar/counter
            if (print)
            {
                // #pragma omp critical
                {

                    time_t currentTime = time(nullptr);
                    double elapsedSeconds = difftime(currentTime, startTime);
                    if (completedIterations > 100)
                    { // start estimating after enough data
                        timePerIteration = elapsedSeconds / completedIterations;
                    }
                    int remainingIterations = numIterations - completedIterations;
                    double remainingSeconds = timePerIteration * remainingIterations;
                    int remainingMinutes = static_cast<int>(remainingSeconds / 60);
                    int elapsedMinutes = static_cast<int>(elapsedSeconds / 60);
                    int remainingHours = 0;
                    if (remainingMinutes >= 60)
                    {
                        remainingHours = remainingMinutes / 60;
                        remainingMinutes %= 60;
                    }

                    int elapsedHours = 0;
                    if (elapsedMinutes >= 60)
                    {
                        elapsedHours = elapsedMinutes / 60;
                        elapsedMinutes %= 60;
                    }

                    double progress = static_cast<double>(completedIterations) / numIterations;
                    int barWidth = 25;
                    int numFilled = static_cast<int>(progress * barWidth);
                    std::string progressBar(numFilled, '#');
                    std::string remainingBar(barWidth - numFilled, '-');

                    std::cout << "Progress: "
                              << static_cast<int>(round(progress * 100.0)) << "% "
                              << "[" << progressBar << remainingBar << "] "
                              << completedIterations << "/" << numIterations;
                    std::cout << " ( ";

                    if (elapsedHours > 0)
                    {
                        std::cout << elapsedHours << " hr ";
                    }

                    std::cout << elapsedMinutes << " min elapsed < ";

                    if (remainingHours > 0)
                    {
                        std::cout << remainingHours << " hr ";
                    }

                    std::cout << remainingMinutes << " min remaining)";

                    std::cout << std::string(10, ' ') << "\r" << std::flush;
                }
            }
        }
    }

    std::cout << std::endl; // Print newline after progress bar is complete

    // Build numpy array from C array
    npy_intp dims[] = {len1, len2};
    PyObject *result = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    PyArrayObject *distancesArr = reinterpret_cast<PyArrayObject *>(result);
    double *data = reinterpret_cast<double *>(PyArray_DATA(distancesArr));

    for (int i = 0; i < len1; ++i)
    {
        for (int j = 0; j < len2; ++j)
        {
            data[i * len2 + j] = res[i][j];
        }
    }

    return result;
}

static PyMethodDef wgwl_methods[] = {
    {"wgwlVec", computeVectorized, METH_VARARGS, "Vectorized version."},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef wgwlModule = {
    PyModuleDef_HEAD_INIT,
    "wgwl",
    "A module to efficiently compute the WGWL kernel.",
    -1,
    wgwl_methods};

PyMODINIT_FUNC PyInit_wgwl(void)
{
    return PyModule_Create(&wgwlModule);
}
