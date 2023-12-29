#include "network/Session.hpp"

#include <cassert>

#include "core/Recursion.hpp"
#include "nodes/OperationNode.hpp"
#include "nodes/PlaceholderNode.hpp"
#include "nodes/ValueNode.hpp"

namespace std
{
template <> struct hash<std::vector<std::shared_ptr<Core::Node>>>
{
	auto operator()(const std::vector<std::shared_ptr<Core::Node>> &nodes) const
	{
		auto seed = nodes.size();
		for (const auto &node : nodes)
		{
			seed ^= std::hash<std::shared_ptr<Core::Node>>{}(node) +
					0x9e3779b9 + (seed << 6) + (seed >> 2);
		}
		return seed;
	}
};
}  // namespace std

namespace Core
{

auto Session::run(PlaceholderMap &map) -> const arma::mat &
{
	const auto &computation_order =
		get_or_compute_computation_order(graph_root);
	for (const auto &node : computation_order)
	{
		const auto type = node->get_type();
		if (type == NodeType::Placeholder)
		{
			auto placeholder = std::static_pointer_cast<PlaceholderNode>(node);
			node->forward(map.get(placeholder->get_identifier()));
		}
		else if (Node::is_operation(type))
		{
			auto operation = std::static_pointer_cast<OperationNode>(node);
			auto consumer_outputs =
				OperationNode::get_consumer_outputs(*operation);
			node->forward(consumer_outputs);
		}
		else if (type == NodeType::Value)
		{
			auto placeholder = std::static_pointer_cast<ValueNode>(node);
			node->forward(placeholder->value);
		}
	}
	return graph_root->value;
}

auto Session::get_or_compute_computation_order(Ref<Node> operation)
	-> std::vector<Ref<Node>> &
{
	std::vector<Ref<Node>> computation_order;
	auto recurse =
		rec([&computation_order](auto &&recurse, Ref<Node> &node) -> void {
			if (Node::is_operation(node->get_type()))
			{
				for (auto &input : node->inputs)
				{
					recurse(input);
				}
			}
			computation_order.push_back(node);
		});

	recurse(operation);

	auto hash = std::hash<std::vector<Ref<Node>>>{}(computation_order);
	if (computation_order_cache.contains(hash))
	{
		return computation_order_cache.at(hash);
	}

	computation_order_cache.emplace(hash, computation_order);
	return computation_order_cache.at(hash);
}

}  // namespace Core