#include <iostream>
#include "lib/tree/tree.hh"
#include "ted.hpp"
#include "tree_helper.hpp"

int main(void)
{
    tree<int> t1 = build_tree<int>("1(2(1,3),3(1))");
    tree<int> t2 = build_tree<int>("1(1(1,2,3))");

    std::vector<tree<int>> subgraphs1{t1, t1, t1};
    std::vector<tree<int>> subgraphs2{t2, t2};

    std::cout << std::to_string(std::hash<std::pair<int, int>>{}(std::make_pair(0, 1))) << std::endl;

    LookupTable lookup;
    auto res = kernel(subgraphs1, subgraphs2, lookup);
    std::cout << res << std::endl;

    return 0;
}