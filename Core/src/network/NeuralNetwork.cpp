#include "network/NeuralNetwork.hpp"
#include "core/Recursion.hpp"
#include "core/Typelist.hpp"
#include "network/Exceptions.hpp"
#include "network/OperationTypelist.hpp"
#include "network/Session.hpp"
#include "nodes/Initializers.hpp"
#include "nodes/Node.hpp"
#include "nodes/PlaceholderNode.hpp"
#include "nodes/ValueNode.hpp"
#include "nodes/operations/AdditionOperation.hpp"
#include "nodes/operations/MatrixMultiplyOperation.hpp"
#include "nodes/operations/NegateOperation.hpp"
#include "nodes/operations/ReLUOperation.hpp"
#include "nodes/operations/ReduceSumOperation.hpp"
#include "nodes/optimizers/SGDOptimizer.hpp"

#include <cassert>
#include <fmt/core.h>
#include <memory>
#include <queue>
#include <type_traits>

namespace Core
{

template <typename OperationType>
	requires(std::is_base_of_v<Node, OperationType>) && requires {
		{
			OperationType::get_operation_type()
		} -> std::same_as<NodeType>;
	}

void register_operation_type()
{
	const auto node_type = OperationType::get_operation_type();
	NeuralNetwork::register_node_type(
		node_type, [](const std::vector<Ref<Node>> &inputs) -> Ref<Node> {
			return std::make_shared<OperationType>(inputs);
		});
}

template <typename Typelist> struct RegisterOperations;

// Specialization for non-empty typelist
template <typename First, typename... Rest>
struct RegisterOperations<Typelist<First, Rest...>>
{
	static void register_all()
	{
		register_operation_type<First>();
		RegisterOperations<Typelist<Rest...>>::register_all();
	}
};

// Specialization for empty typelist (base case)
template <> struct RegisterOperations<Typelist<>>
{
	static void register_all()
	{
	}
};

NeuralNetwork::NeuralNetwork(const std::vector<LayerDefinition> &definitions,
							 u32 inputs)
	: input_units(inputs)
{
	RegisterOperations<OperationTypes>::register_all();
	build_network(definitions);
}

void NeuralNetwork::forward()
{
	forward(activation_root);
}

void NeuralNetwork::forward(Ref<Node> &root)
{
	Session session{root};
	session.run(placeholder_map);
}

auto NeuralNetwork::build_network(
	const std::vector<LayerDefinition> &definitions) -> void
{
	using namespace Core::Initialization;
	using Xavier = Core::Initialization::Initializer<Method::Xavier>;
	using RefVector = std::vector<std::shared_ptr<Node>>;

	auto placeholder_x = make_placeholder("x");
	auto placeholder_loss = make_placeholder("c");

	placeholder_map.map_value("x", placeholder_x);
	placeholder_map.map_value("c", placeholder_loss);

	arma::mat weight;
	arma::vec bias;
	Xavier::initialize(weight, input_units, definitions[0].units);
	Xavier::initialize(bias, definitions[0].units);

	if (definitions.at(0).units == 0)
	{
		throw IncompatibleLayerDefinitionException(
			"Layer definition must have at least one unit");
	}

	auto first_weight = make_value(weight);
	auto first_bias = make_value(bias);

	auto matmul =
		make_operation<MatrixMultiplyOperation>(placeholder_x, first_weight);
	auto add = make_operation<AdditionOperation>(matmul, first_bias);

	auto operation =
		activations_map.at(definitions[0].activation)(RefVector{add});

	for (auto i = 1ULL; i < definitions.size(); ++i)
	{
		auto &definition = definitions[i];

		if (definition.units == 0)
		{
			throw IncompatibleLayerDefinitionException(
				"Layer definition must have at least one unit");
		}

		arma::mat other_weight;
		arma::vec other_bias;
		Xavier::initialize(other_weight, definitions[i - 1].units,
						   definitions[i].units);
		Xavier::initialize(other_bias, definitions[i].units);
		auto first_other_weight = make_value(other_weight);
		auto first_other_bias = make_value(other_bias);

		auto other_matmul = make_operation<MatrixMultiplyOperation>(
			operation, first_other_weight);
		auto other_add =
			make_operation<AdditionOperation>(other_matmul, first_other_bias);
		operation = activations_map.at(definition.activation)({other_add});
	}

	activation_root = operation;
	// Following is what we need to implement:
	// J = negative(reduce_sum(reduce_sum(multiply(c, log(p)), axis=1)))
	auto log = make_operation<LogOperation>(activation_root);
	auto multiply = make_operation<MultiplyOperation>(placeholder_loss, log);
	auto reduce_sum = make_operation<ReduceSumOperation>(multiply, 1);
	auto reduce_other_dimension =
		make_operation<ReduceSumOperation>(reduce_sum);
	auto negative = make_operation<NegateOperation>(reduce_other_dimension);
	loss_root = negative;

	optimizer = std::make_shared<SGDOptimizer>(loss_root, 0.001);
}

auto NeuralNetwork::register_node_type(NodeType type, NodeTypeFactory &&factory)
	-> void
{
	if (activations_map.contains(type))
	{
		fmt::print("Node type {} already registered\n", to_string(type));
		return;
	}

	activations_map[type] = factory;
}

auto NeuralNetwork::pretty_print() const -> void
{
	using namespace std::string_view_literals;

	fmt::println("Computation order:");
	auto depth = 0ULL;
	auto print_node_information = rec([](auto &&print_node_information,
										 auto &node, auto dep) -> void {
		for (auto i = 0ULL; i < dep; i++)
		{
			fmt::print(" ");
		}
		if (node->get_type() == NodeType::Value)
		{
			auto cast = std::static_pointer_cast<ValueNode>(node);
			const auto &&[mean, stdev] = node->get_weight_statistics();
			const auto &matrix = cast->value;
			fmt::print("ValueNode: {} - [{}, {}]. Mean: {}, stdev: {}\n",
					   matrix.n_cols == 1 ? "Vector"sv : "Matrix"sv,
					   matrix.n_rows, matrix.n_cols, mean, stdev);
		}

		if (node->get_type() == NodeType::Placeholder)
		{
			auto cast = std::static_pointer_cast<PlaceholderNode>(node);
			fmt::print("PlaceholderNode: Name - \"{}\"\n",
					   cast->get_identifier());
		}

		if (Node::is_operation(node->get_type()))
		{
			auto cast = std::static_pointer_cast<OperationNode>(node);

			fmt::print("Operation: {}. Inputs:\n", to_string(cast->get_type()));
			for (const auto &input : cast->inputs)
			{
				print_node_information(input, dep + 1);
			}
		}
	});
	print_node_information(activation_root, depth);
	fmt::println("");
}

auto NeuralNetwork::validate_matching_input_size(const arma::mat &vector)
	-> void
{
	if (vector.n_cols != input_units)
	{
		throw IncompatibleInputException(
			fmt::format("Input vector has {} rows and {} columns, but network "
						"expects -1 rows and {} columns",
						vector.n_rows, vector.n_cols, input_units));
	}
}

auto NeuralNetwork::predict(const arma::vec &vector) -> arma::mat
{
	validate_matching_input_size(vector);

	placeholder_map.get("x") = vector;
	forward();
	return activation_root->value;
}

auto NeuralNetwork::predict(const arma::mat &matrix) -> arma::mat
{
	validate_matching_input_size(matrix);

	placeholder_map.get("x") = matrix;
	forward();
	return activation_root->value;
}

auto NeuralNetwork::train() -> void
{
	Session session{optimizer};
	const auto one_hot_encoded_6_by_2 = arma::ones<arma::mat>(6, 2);

	placeholder_map.get("c") = one_hot_encoded_6_by_2;
	session.run(placeholder_map);
}

}  // namespace Core
