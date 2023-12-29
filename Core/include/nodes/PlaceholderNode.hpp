#pragma once

#include "nodes/Node.hpp"

namespace Core
{

class PlaceholderNode : public Node
{
  public:
	~PlaceholderNode() override = default;
	explicit PlaceholderNode(std::string_view id) : Node(), identifier(id)
	{
	}

	auto get_type() const -> NodeType override
	{
		return NodeType::Placeholder;
	}
	auto get_units() const -> std::uint32_t override
	{
		return 0;
	}
	auto get_weight_statistics() const -> WeightStatistics override
	{
		return {};
	}

	void forward(arma::mat &val) override
	{
		this->value = val;
	}

	void backward() override
	{
		// No operation, as placeholders do not contribute to gradients
	}

	auto get_identifier() const -> std::string
	{
		return identifier;
	}

  private:
	std::string identifier;
};

inline auto make_placeholder(std::string_view id) -> Ref<Node>
{
	return std::make_shared<PlaceholderNode>(id);
}

}  // namespace Core
