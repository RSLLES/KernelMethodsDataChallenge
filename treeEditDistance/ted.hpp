#ifndef TED
#define TED

#include <iostream>
#include <unordered_map>
#include "lib/tree/tree.hh"
#include "lib/hungarian/hungarian.h"
#include "lib/wassertein/Wasserstein.hh"

#define COST_RENAME 1
#define COST_MOVE 1

#define THRESHOLD 100000000
#define NB_TO_REMOVE 50000000

using hash_type = uint64_t;

using LookupTable = std::unordered_map<hash_type, float>;

inline hash_type merge_hash(const hash_type &a, const hash_type &b)
{
    return a >= b ? a * a + a + b : a + b * b;
}

inline hash_type sim_merge_hash(const hash_type &a, const hash_type &b)
{
    return merge_hash(a + b, a * b);
}

template <typename T>
std::pair<size_t, hash_type> count_and_hash_nodes_below(const tree<T> &t, const typename tree<T>::pre_order_iterator &it, char depth = 0)
{
    size_t count = 1; // start with 1 to include the current node

    hash_type hash = static_cast<hash_type>(*(it));
    for (auto child_it = t.begin(it); child_it != t.end(it); ++child_it)
    {
        auto [c, h] = count_and_hash_nodes_below(t, child_it, depth + 1);
        hash = sim_merge_hash(hash, h);
        count += c;
    }

    hash = merge_hash(hash, depth);
    return {count, hash};
}

template <class T>
double TreeEditDistance(tree<T> &t1, tree<T> &t2, LookupTable &lookup)
{
    HungarianAlgorithm HungAlgo;
    return ted(t1, t1.begin(), t2, t2.begin(), HungAlgo, lookup);
}

template <class T, typename iter>
double ted(tree<T> &t1, const iter &i1, tree<T> &t2, const iter &i2, HungarianAlgorithm &HungAlgo, LookupTable &lookup)
{
    // If there is a dummy
    if (*i1 == -1 && *i2 == -1)
    {
        return 0.0;
    }

    // hash trees
    auto [c1, h1] = count_and_hash_nodes_below(t1, i1);
    auto [c2, h2] = count_and_hash_nodes_below(t2, i2);
    hash_type h = sim_merge_hash(h1, h2);

    double cost = 0.0;
#pragma omp critical
    {
        auto it = lookup.find(h);
        if (it != lookup.end())
        {
            cost = static_cast<double>(it->second); // Return precomputed TED value.
        }
    }
    if (cost != 0.0)
    {
        return cost;
    }

    if (*i1 == -1 && *i2 != -1)
    {
        return COST_MOVE * c2;
    }
    if (*i1 != -1 && *i2 == -1)
    {
        return COST_MOVE * c1;
    }

    cost = *i1 == *i2 ? 0.0 : COST_RENAME;

    // If both have no childs
    size_t size1 = t1.number_of_children(i1), size2 = t2.number_of_children(i2);
    if (size1 == 0 || size2 == 0)
    {
        return cost;
    }

    // They both have children. Computing adjencyMatrix.
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
            row[i] = ted(t1, it1, t2, it2, HungAlgo, lookup);
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
    double solve = cost + HungAlgo.Solve(costMatrix, assignment);
#pragma omp critical
    {
        if (lookup.size() > THRESHOLD)
        {
            std::cout << "Cleaning ...";
            auto start_it = lookup.begin();
            auto end_it = start_it;
            std::advance(end_it, NB_TO_REMOVE);
            lookup.erase(start_it, end_it);
        }
        lookup[h] = static_cast<float>(solve);
    }
    return solve;
}

template <typename T>
double kernel(std::vector<tree<T>> subtrees1, std::vector<tree<T>> subtrees2, LookupTable &lookup)
{
    const size_t len1 = subtrees1.size(), len2 = subtrees2.size();
    std::vector<double> distances(len1 * len2);

    for (int i = 0; i < len1; ++i)
    {
        for (int j = 0; j < len2; ++j)
        {
            distances[i * len2 + j] = TreeEditDistance(subtrees1[i], subtrees2[j], lookup);
            // std::cout << distances[i * len2 + j] << " | ";
        }
    }

    // Compute wassertein distance
    std::vector<double> weights1(len1, 1.0);
    std::vector<double> weights2(len2, 1.0);
    wasserstein::EMD<double, wasserstein::DefaultEvent, wasserstein::DefaultPairwiseDistance> emd(1.0, 1.0, true);
    emd.ground_dists() = std::move(distances);
    return emd(weights1, weights2);
}

#endif