#pragma once

#include "nodes/OperationNode.hpp"
#include <cassert>

namespace Core
{

class MatrixMultiplyOperation : public OperationNode
{
  public:
	~MatrixMultiplyOperation() override = default;
	using OperationNode::OperationNode;

	auto forward(const std::vector<arma::mat> &consumer_outputs)
		-> void override;

	auto backward() -> void override
	{
	}

	auto propagate_gradient(const arma::mat &input) -> std::vector<arma::mat> override;

	auto get_type() const -> NodeType override
	{
		return NodeType::MatrixMultiply;
	}
	static auto get_operation_type() -> NodeType
	{
		return NodeType::MatrixMultiply;
	}
};

}  // namespace Core