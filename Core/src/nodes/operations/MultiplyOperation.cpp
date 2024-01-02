#include "nodes/operations/MultiplyOperation.hpp"
#include "nodes/Node.hpp"

namespace Core
{

auto MultiplyOperation::forward(const std::vector<arma::mat> &consumer_outputs)
	-> void
{
	arma::mat broadcastedA, broadcastedB;
	const auto &left = consumer_outputs.at(0);
	const auto &right = consumer_outputs.at(1);

	broadcast_matrices(left, right, broadcastedA, broadcastedB);
	value = broadcastedA % broadcastedB;
}

std::vector<arma::mat> MultiplyOperation::propagate_gradient(
	const arma::mat &input)
{
	// Value field is an arma::mat
	const auto &A = this->inputs.at(0)->value;
	const auto &B = this->inputs.at(1)->value;

	arma::mat expandedA = arma::repmat(A, 1, input.n_cols);
	arma::mat expandedB = arma::repmat(B, 1, input.n_cols);

	auto input_repmat_a =
		arma::repmat(input, expandedA.n_rows, expandedA.n_cols);
	auto input_repmat_b =
		arma::repmat(input, expandedB.n_rows, expandedB.n_cols);

	fmt::print("MultiplyOperation Gradients\n");

	return {input_repmat_a % expandedB, input_repmat_b % expandedA};
}

void MultiplyOperation::broadcast_matrices(const arma::mat &A,
										   const arma::mat &B,
										   arma::mat &broadcast_a,
										   arma::mat &broadcast_b)
{
	if (A.n_rows == B.n_rows && A.n_cols == B.n_cols)
	{
		// No broadcasting needed
		broadcast_a = A;
		broadcast_b = B;
	}
	else if (A.n_rows == B.n_rows && B.n_cols == 1)
	{
		// Broadcast B across columns
		broadcast_a = A;
		broadcast_b = arma::repmat(B, 1, A.n_cols);
	}
	else if (A.n_cols == B.n_cols && B.n_rows == 1)
	{
		// Broadcast B across rows
		broadcast_a = A;
		broadcast_b = arma::repmat(B, A.n_rows, 1);
	}
	else if (A.n_cols == 1 && A.n_rows == B.n_rows)
	{
		// Broadcast A across columns
		broadcast_a = arma::repmat(A, 1, B.n_cols);
		broadcast_b = B;
	}
	else if (A.n_rows == 1 && A.n_cols == B.n_cols)
	{
		// Broadcast A across rows
		broadcast_a = arma::repmat(A, B.n_rows, 1);
		broadcast_b = B;
	}
	else
	{
		throw std::runtime_error("Incompatible shapes for broadcasting");
	}
}

}  // namespace Core