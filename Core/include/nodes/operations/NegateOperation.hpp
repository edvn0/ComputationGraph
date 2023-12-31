#pragma once

#include "nodes/OperationNode.hpp"
#include <cassert>

namespace Core
{

class NegateOperation : public OperationNode
{
  public:
	~NegateOperation() override = default;
	using OperationNode::OperationNode;

	auto forward(const std::vector<arma::mat> &consumer_outputs)
		-> void override;

	auto backward() -> void override
	{
	}

	auto propagate_gradient(const arma::mat &input)
		-> std::vector<arma::mat> override;

	auto get_type() const -> NodeType override
	{
		return NodeType::Negate;
	}
	static auto get_operation_type() -> NodeType
	{
		return NodeType::Negate;
	}
};

}  // namespace Core