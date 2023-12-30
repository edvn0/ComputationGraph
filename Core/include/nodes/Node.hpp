#pragma once

#include <armadillo>
#include <memory>
#include <string>
#include <utility>

#include <fmt/core.h>

#include "core/Types.hpp"

namespace Core
{

using WeightStatistics = std::pair<double, double>;

enum class NodeType : std::uint8_t
{
	Activation,
	Addition,
	BatchNormalization,
	Convolution,
	Dense,
	Dropout,
	ELU,
	Flatten,
	LeakyReLU,
	Log,
	MatrixMultiply,
	Multiply,
	Negate,
	None,
	Operation,
	Placeholder,
	Pooling,
	ReduceSum,
	ReLU,
	Reshape,
	SELU,
	SGD,
	Sigmoid,
	Softmax,
	SoftPlus,
	SoftSign,
	TanH,
	Value,
};

auto to_string(NodeType) -> std::string;

class Node : public std::enable_shared_from_this<Node>
{
  public:
	virtual ~Node() = default;

	// Typical for Value and Placeholder nodes.
	virtual void forward(arma::mat &)
	{
		fmt::println("Node::forward(arma::mat &tensor) called");
	};

	virtual void forward()
	{
		fmt::println("Node::forward(arma::mat &) called");
	};

	// Typical for Operation nodes.
	virtual void forward(const std::vector<arma::mat> &){};

	virtual void backward() = 0;
	virtual auto propagate_gradient(const arma::mat &input) -> arma::mat
	{
		return input;
	};

	virtual auto get_type() const -> NodeType = 0;
	virtual auto get_units() const -> std::uint32_t = 0;
	virtual auto get_weight_statistics() const -> WeightStatistics = 0;

	static auto is_operation(NodeType type) -> bool;
	static auto is_optimizer(NodeType type) -> bool;

	std::vector<Ref<Node>> inputs;
	std::vector<Weak<Node>> consumers;
	arma::mat value{};

  protected:
	explicit Node(const std::vector<Ref<Node>> &node_inputs = {},
				  const arma::mat &matrix = {})
		: inputs(node_inputs), value(matrix)
	{
		for (auto &input : this->inputs)
		{
			input->consumers.push_back(this->weak_from_this());
		}
	}
};

}  // namespace Core
