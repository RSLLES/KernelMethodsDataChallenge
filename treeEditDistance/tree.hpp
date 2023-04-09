#ifndef TREE
#define TREE

#include <algorithm>
#include <iostream>
#include <map>
#include <vector>

using namespace std;

template <typename X>
vector<X> emit(const vector<X> x, string s)
{
	cout << s << ": ";
	for (auto xx : x)
	{
		cout << xx << " ";
	}
	cout << "\n";
	return x;
}
template <typename X>
X emit(const X x, string s)
{
	cout << s << ": " << x << "\n";
	return x;
}
template <typename X>
vector<X> emit(const vector<X> x)
{
	for (auto xx : x)
	{
		cout << xx << " ";
	}
	cout << "\n";
	return x;
}
template <typename X>
X emit(const X x)
{
	cout << x << "\n";
	return x;
}

template <typename X>
vector<X> except(const vector<X> x, const vector<X> y)
{
	vector<X> r;
	r.reserve(x.size());
	auto begy = begin(y), endy = end(y);
	for (auto xx : x)
		if (find(begy, endy, xx) == endy)
			r.push_back(xx); // XXX use copy here
	return r;
}
template <typename X>
vector<X> except(const vector<X> x, const X y)
{
	vector<X> r;
	r.reserve(x.size());
	for (auto xx : x)
		if (xx != y)
			r.push_back(xx);
	return r;
}

template <typename X>
// index X with y, and then X again with the result, until it returns the same thing twice
// returns all steps
// i.e., [ x[y], x[x[y]], x[x[x[y]]], ...
vector<X> exhaust(const vector<X> x, X y)
{
	X i = y, last;
	vector<X> r;
	r.reserve(x.size());
	while (1)
	{
		last = i;
		i = emit(x[i]);
		if (i == last)
			return r;
		else
			r.push_back(i);
	};
	return r;
}

template <typename X>
class Tree
{
public:
	vector<X> x;
	vector<int> p;
	vector<X> operator[](vector<int> path)
	{
		vector<X> r;
		r.reserve(path.size());
		for (auto p : path)
			r.push_back(x[p]);
		return r;
	}
	int adopt(const int parent, const int child)
	{
		p[child] = parent;
		return child;
	}
	int insert(const int parent, const X item)
	{
		x.push_back(item);
		p.push_back(parent);
		return x.size() - 1;
	}
	int parent(const int child)
	{
		return p[child];
	}
	vector<int> childs(const int parent)
	{
		vector<int> childs;
		int start = (parent == 0) ? 1 : 0;
		for (int i = start; i < p.size(); ++i)
		{
			if (p[i] == parent)
			{
				childs.push_back(i);
			}
		}
		return childs;
	}
	size_t nb_childs(const int parent)
	{
		auto all_childs = childs(parent);
		return all_childs.size();
	}
	bool has_childs(const int parent)
	{
		vector<int> childs;
		int start = (parent == 0) ? 1 : 0;
		for (int i = start; i < p.size(); ++i)
		{
			if (p[i] == parent)
			{
				return true;
			}
		}
		return false;
	}
	vector<int> path(const int child)
	{
		return exhaust(p, child);
	}
	void remove_last(size_t n)
	{
		if (p.size() > n)
		{
			p.erase(p.begin() + (p.size() - n), p.end());
			x.erase(x.begin() + (x.size() - n), x.end());
		}
		else
		{
			// If vector has less than n number of elements,
			// then delete all elements
			std::cout << "Warning : graph erased." << std::endl;
			p.erase(p.begin(), p.end());
			x.erase(x.begin(), x.end());
		}
	}
	size_t size()
	{
		return p.size();
	}
};

template <typename X>
ostream &operator<<(ostream &os, const Tree<X> T)
{
	int n = 0;
	for (int n = 0; n < T.x.size(); ++n)
	{
		os << n << ". " << T.x[n] << " -> " << T.p[n] << ".\n";
	}
	return os;
}

#endif