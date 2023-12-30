#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <unordered_map>

#include "network/Exceptions.hpp"
#include "network/LayerDefinition.hpp"
#include "network/NeuralNetwork.hpp"

template <class T> using TestPtr = std::unique_ptr<T>;

auto create_random_arma_matrix = [](auto rows, auto cols) {
	arma::mat matrix(rows, cols);
	matrix.randn();
	return matrix;
};

TEST_CASE("Neural network")
{

	SECTION("Creation works like intended")
	{
		TestPtr<Core::NeuralNetwork> network{nullptr};
		auto layer_definitions = std::vector{
			Core::LayerDefinition{
				10,
				Core::NodeType::ReLU,
			},
			Core::LayerDefinition{
				30,
				Core::NodeType::Softmax,
			},
		};

		REQUIRE_NOTHROW(network = std::make_unique<Core::NeuralNetwork>(
							layer_definitions, 7),
						"Creation failed");
	}

	SECTION("Creation should not work")
	{
		TestPtr<Core::NeuralNetwork> network{nullptr};

		auto layer_definitions = std::vector{
			Core::LayerDefinition{
				0,
				Core::NodeType::ReLU,
			},
			Core::LayerDefinition{
				0,
				Core::NodeType::Softmax,
			},
		};

		REQUIRE_THROWS_AS(network = std::make_unique<Core::NeuralNetwork>(
							  layer_definitions, 7),
						  Core::IncompatibleLayerDefinitionException);
	}

	SECTION("Correct predicted output dimensions")
	{
		TestPtr<Core::NeuralNetwork> network{nullptr};
		auto layer_definitions = std::vector{
			Core::LayerDefinition{
				10,
				Core::NodeType::ReLU,
			},
			Core::LayerDefinition{
				30,
				Core::NodeType::Softmax,
			},
		};

		REQUIRE_NOTHROW(network = std::make_unique<Core::NeuralNetwork>(
							layer_definitions, 7),
						"Creation failed");

		auto input = create_random_arma_matrix(10, 7);
		auto output = network->predict(input);

		REQUIRE(output.n_rows == 10);
		REQUIRE(output.n_cols == 30);
	}

	SECTION("Fail on incorrect predicted output")
	{
		TestPtr<Core::NeuralNetwork> network{nullptr};
		auto layer_definitions = std::vector{
			Core::LayerDefinition{
				10,
				Core::NodeType::ReLU,
			},
			Core::LayerDefinition{
				30,
				Core::NodeType::Softmax,
			},
		};

		REQUIRE_NOTHROW(network = std::make_unique<Core::NeuralNetwork>(
							layer_definitions, 7),
						"Creation failed");

		REQUIRE_THROWS_AS(network->predict(create_random_arma_matrix(10, 10)),
						  Core::IncompatibleInputException);
	}
}