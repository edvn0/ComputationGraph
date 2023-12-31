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
	auto get_units() const -> u32 override
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

	static auto extract_matrix_unsafe(const Ref<Node>& node)
	{
		return node.get()->value;
	}
};

inline auto make_value(const arma::mat &tensor) -> Ref<Node>
{
	auto output = std::make_shared<ValueNode>(tensor);
	output->add_consumers();
	return output;
}

}  // namespace Core
