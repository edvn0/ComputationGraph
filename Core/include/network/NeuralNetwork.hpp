#pragma once

#include "network/LayerDefinition.hpp"
#include "network/Session.hpp"
#include "nodes/Node.hpp"
#include <memory>
#include <vector>

namespace Core
{

class NeuralNetwork
{
	using NodeTypeFactory =
		decltype(+[](const std::vector<std::shared_ptr<Node>> &)
					 -> std::shared_ptr<Node> { return nullptr; });
	static inline std::unordered_map<NodeType, NodeTypeFactory> activations_map;

  public:
	explicit NeuralNetwork(const std::vector<LayerDefinition> &definitions,
						   std::uint32_t inputs = 0);
	~NeuralNetwork() = default;
	void forward();
	void forward(Ref<Node> &);
	void pretty_print() const;
	auto predict(const arma::vec &) -> arma::mat;
	auto predict(const arma::mat &) -> arma::mat;

	auto train() -> void;

	static auto register_node_type(NodeType type, NodeTypeFactory &&factory)
		-> void;

  private:
	Ref<Node> activation_root;
	Ref<Node> loss_root;
	Ref<Node> optimizer;

	std::uint32_t input_units{};
	auto validate_matching_input_size(const arma::mat &) -> void;

	PlaceholderMap placeholder_map;

	void build_network(const std::vector<LayerDefinition> &definitions);
};

}  // namespace Core
