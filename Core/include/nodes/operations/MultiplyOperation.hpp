#pragma once

#include "nodes/OperationNode.hpp"
#include <cassert>

namespace Core
{

class MultiplyOperation : public OperationNode
{
  public:
	~MultiplyOperation() override = default;
	using OperationNode::OperationNode;

	auto forward(const std::vector<arma::mat> &consumer_outputs)
		-> void override;
	auto backward() -> void override
	{
	}

	auto propagate_gradient(const arma::mat &input) -> std::vector<arma::mat> override;

	auto get_type() const -> NodeType override
	{
		return NodeType::Multiply;
	}
	static auto get_operation_type() -> NodeType
	{
		return NodeType::Multiply;
	}

  private:
	void broadcast_matrices(const arma::mat &, const arma::mat &, arma::mat &,
							arma::mat &);
};

}  // namespace Core