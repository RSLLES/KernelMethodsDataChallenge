#ifndef TED
#define TED

#include <iostream>
#include "lib/tree/tree.hh"
#include "hungarian/hungarian.h"

#define COST_RENAME 1
#define COST_MOVE 1

template <class t>
void display_matrix(const std::vector<std::vector<t>> &matrix)
{
    for (auto row : matrix)
    {
        for (auto cell : row)
        {
            std::cout << cell << " ";
        }
        std::cout << std::endl;
    }
}

template <typename T>
size_t count_nodes_below(const tree<T> &t, const typename tree<T>::pre_order_iterator &it)
{
    size_t count = 1; // start with 1 to include the current node
    for (auto child_it = t.begin(it); child_it != t.end(it); ++child_it)
    {
        count += count_nodes_below(t, child_it);
    }
    return count;
}

template <class T>
double TreeEditDistance(tree<T> &t1, tree<T> &t2)
{
    HungarianAlgorithm HungAlgo;
    return ted(t1, t1.begin(), t2, t2.begin(), HungAlgo);
}

template <class T, typename iter>
double ted(tree<T> &t1, const iter &i1, tree<T> &t2, const iter &i2, HungarianAlgorithm &HungAlgo)
{
    // If there is a dummy
    if (*i1 == -1 && *i2 == -1)
    {
        return 0.0;
    }
    if (*i1 == -1 && *i2 != -1)
    {
        return COST_MOVE * count_nodes_below(t2, i2);
    }
    if (*i1 != -1 && *i2 == -1)
    {
        return COST_MOVE * count_nodes_below(t1, i1);
    }

    double cost = *i1 == *i2 ? 0.0 : COST_RENAME;

    // If both have no childs
    if (t1.number_of_children(i1) == 0 || t1.number_of_children(i1) == 0)
    {
        if (!(t1.number_of_children(i1) == 0 && t1.number_of_children(i1) == 0))
        {
            std::cerr << "ERROR : Only nodes within the same level can be compared." << std::endl;
            return 0.0;
        }
        return cost;
    }

    // They both have children. Computing adjencyMatrix.
    size_t size1 = t1.number_of_children(i1), size2 = t2.number_of_children(i2);
    auto last1 = t1.child(i1, size1 - 1), last2 = t2.child(i2, size2 - 1);
    size_t size = max(size1, size2) + 1;

    for (size_t s = size1; s < size; ++s)
        t1.append_child(i1, -1);
    for (size_t s = size2; s < size; ++s)
        t2.append_child(i2, -1);

    vector<vector<double>> costMatrix(size, std::vector<double>(size));
    auto it1 = t1.child(i1, 0);
    for (auto &row : costMatrix)
    {
        auto it2 = t2.child(i2, 0);
        for (int i = 0; i < row.size(); ++i)
        {
            row[i] = ted(t1, it1, t2, it2, HungAlgo);
            ++it2;
        }
        ++it1;
    }

    // display_matrix<double>(costMatrix);
    // std::cout << "-----" << std::endl;

    // Delete dummies
    t1.erase_right_siblings(last1);
    t2.erase_right_siblings(last2);

    // Compute cost
    vector<int> assignment;
    return cost + HungAlgo.Solve(costMatrix, assignment);
}

#endif