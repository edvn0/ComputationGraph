#pragma once

#include "core/PointerMap.hpp"
#include "nodes/Node.hpp"
#include "nodes/PlaceholderNode.hpp"

#include <cstddef>
#include <unordered_map>

namespace Core
{

using PlaceholderMap = PointerMap<PlaceholderNode, arma::mat>;

class Session
{
	static inline std::unordered_map<std::size_t, std::vector<Ref<Node>>>
		computation_order_cache{};

  public:
	explicit Session(Ref<Node> graph) : graph_root(graph)
	{
	}
	~Session()
	{
		computation_order_cache.clear();
	}
	auto run(PlaceholderMap &) -> const arma::mat &;

	auto get_current_computation_order() -> std::vector<Ref<Node>>
	{
		return get_or_compute_computation_order(graph_root);
	}

  private:
	Ref<Node> graph_root;

	/**
	 * Retrieves or computes the computation order for a given node.
	 *
	 * This function returns a reference to a vector of nodes that represents
	 * the computation order for the given node. If the computation order has
	 * already been computed, it is retrieved. Otherwise, it is computed and
	 * stored for future use.
	 *
	 * @param node The node for which to retrieve or compute the computation
	 * order.
	 * @return A reference to a vector of nodes representing the computation
	 * order.
	 */
	auto get_or_compute_computation_order(Ref<Node> node)
		-> std::vector<Ref<Node>>;
};

}  // namespace Core