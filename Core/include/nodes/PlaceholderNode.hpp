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
	auto output = std::make_shared<PlaceholderNode>(id);
	output->add_consumers();
	return output;
}

}  // namespace Core
