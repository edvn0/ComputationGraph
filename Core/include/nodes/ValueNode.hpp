#pragma once

#include "nodes/Node.hpp"

namespace Core
{

class ValueNode : public Node
{
  public:
	~ValueNode() override = default;
	explicit ValueNode(const arma::mat &input_value) : Node({}, input_value)
	{
	}

	auto get_type() const -> NodeType override
	{
		return NodeType::Value;
	}
	auto get_units() const -> std::uint32_t override
	{
		return 0;
	}
	auto get_weight_statistics() const -> WeightStatistics override;

	void forward(arma::mat &tensor) override
	{
		this->value = tensor;
	}

	void backward() override
	{
		// TODO: Implement
	}

	auto set_value(const arma::mat &tensor)
	{
		value = tensor;
	}
};

inline auto make_value(const arma::mat &tensor) -> Ref<Node> {
  return std::make_shared<ValueNode>(tensor);
}

}  // namespace Core
