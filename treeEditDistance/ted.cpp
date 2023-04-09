#include <iostream>
#include "Hungarian.h"
#include "tree.hpp"
#include "ted.h"

vector<vector<double>> adjency_matrix(Tree<int> &t1, int root1, Tree<int> &t2, int root2)
{
    // if (!t1.has_childs(root1) || !t2.has_childs(root2))
    // {
    //     std::cout << "Error childs" << std::endl;
    //     return vector<vector<double>>();
    // }

    // Get Size
    const size_t size1 = t1.nb_childs(root1), size2 = t2.nb_childs(root2);
    const size_t size = max(size1, size2) + 1;

    // Add dummies
    vector<int> dummies1(size - t1.nb_childs(root1));
    for (int i = 0; i < dummies1.size(); i++)
    {
        dummies1[i] = t1.insert(root1, -1);
    }
    vector<int> dummies2(size - t2.nb_childs(root2));
    for (int i = 0; i < dummies2.size(); i++)
    {
        dummies2[i] = t2.insert(root2, -1);
    }

    const vector<int> childs1 = t1.childs(root1);
    const vector<int> childs2 = t2.childs(root2);

    // Create Matrix
    vector<vector<double>> costMatrix(size, vector<double>(size));
    for (int i = 0; i < childs1.size(); ++i)
    {
        const int child1 = childs1[i];
        for (int j = 0; j < childs1.size(); ++j)
        {
            const int child2 = childs2[j];
            int cost = 0;
            if (t1.x[child1] == -1 && t2.x[child2] == -1)
            {
                cost = 0;
            }
            else if (t1.x[child1] == -1 && t2.x[child2] != -1)
            {
                cost = 1;
            }
            else if (t1.x[child1] != -1 && t2.x[child2] == -1)
            {
                cost = 1;
            }
            else if (t1.has_childs(child1) || t1.has_childs(child1))
            {
                if (!(t1.has_childs(child1) && t1.has_childs(child1)))
                {
                    std::cout << "ERROR : should compare child at the same level" << std::endl;
                    return vector<vector<double>>();
                }

                std::cout << "Exploring " << std::endl;
                cost = ted(t1, child1, t2, child2);
            }
            else if (t1.x[child1] == t2.x[child2])
            {
                cost = 0;
            }
            else
            {
                cost = 1;
            }

            costMatrix[i][j] = cost;
        }
    }

    // Remove dummies
    t1.remove_last(dummies1.size());
    t2.remove_last(dummies2.size());

    return costMatrix;
}

double ted(Tree<int> &t1, int root1, Tree<int> &t2, int root2)
{
    auto costMatrix = adjency_matrix(t1, root1, t2, root2);
    HungarianAlgorithm HungAlgo;
    vector<int> assignment;
    const double cost = root1 == root2 ? 0.0 : 1.0;
    return HungAlgo.Solve(costMatrix, assignment) + cost;
}