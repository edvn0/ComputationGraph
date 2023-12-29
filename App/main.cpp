#include "network/NeuralNetwork.hpp"
#include <iostream>

template <std::size_t Size>
auto generate_random_vector = []() -> arma::vec {
	arma::colvec v(Size);
	v.imbue([]() {
		return arma::randu();
	});	 // Fill with uniform random numbers [0,1]
	return v;
};

template <std::size_t N> std::array<arma::vec, N> generate_random_vectors()
{
	std::array<arma::vec, N> array;
	for (auto &vec : array)
	{
		vec = arma::randu<arma::vec>(2);  // Each vec is of size 2
	}
	return array;
}

int main()
{
	using namespace Core;

	std::vector<LayerDefinition> layers = {
		{3, NodeType::ReLU},
		{2, NodeType::Softmax},
	};

	NeuralNetwork nn(layers, 2);
	nn.pretty_print();
	// Set input and perform forward pass
	// ...

	constexpr auto count = 100ULL;
	constexpr auto print_interval = count / 10;
	auto data = generate_random_vectors<10 * count>();

	arma::mat X = arma::randu<arma::mat>(6, 2);
	// Vector to store the duration of each iteration
	std::vector<double> iteration_durations_ns;
	iteration_durations_ns.reserve(data.size() * count);

	for (auto i = 0; i < count; i++)
	{

		for (const auto &d : data)
		{
			auto start = std::chrono::high_resolution_clock::now();
			const auto output = nn.predict(X);
			auto end = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double, std::nano> elapsed = end - start;
			iteration_durations_ns.push_back(elapsed.count());
#ifdef CG_DEBUG
			output.print();
#endif
		}

		// Print message and statistics every 100 iterations
		if ((i + 1) % print_interval == 0)
		{
			arma::vec durations(iteration_durations_ns.data(),
								iteration_durations_ns.size());
			double mean = arma::mean(durations);
			double stddev = arma::stddev(durations);

			fmt::print(
				"Finished iteration {}/{} - Mean time thus far: {:.3f} ns, "
				"Stddev: {:.3f} ns\n",
				i + 1, count, mean, stddev);
		}
	}

	// Convert durations to an arma::vec for statistical computations
	arma::vec durations(iteration_durations_ns);

	// Compute statistics
	double mean = arma::mean(durations);
	double stddev = arma::stddev(durations);

	double mean_ms = mean / 1000000.0;
	double stddev_ms = stddev / 1000000.0;

	fmt::print("Iteration statistics: Mean - {:.6f} ns ({:.6f} ms), Stddev - "
			   "{:.6f} ns ({:.6f} ms)\n",
			   mean, mean_ms, stddev, stddev_ms);

	const auto output = nn.predict(data.at(2));
	output.print();

	return 0;
}
