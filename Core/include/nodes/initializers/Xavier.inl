#include <armadillo>

namespace Core::Initialization
{

template <> struct Initializer<Method::Xavier>
{
  public:
	static void initialize(arma::mat &weights, std::uint32_t input_units,
						   std::uint32_t output_units)
	{
		const auto variance = 2.0 / (input_units + output_units);

		// resize
		weights.resize(input_units, output_units);
		weights.imbue([&]() { return arma::randn() * std::sqrt(variance); });
	}

	static void initialize(arma::vec &weights, std::uint32_t input_units)
	{
		const auto variance = 2.0 / (input_units);
		// resize
		weights.resize(input_units);
		weights.imbue([&]() { return arma::randn() * std::sqrt(variance); });
	}
};

}  // namespace Core::Initialization