#include <armadillo>

namespace Core::Initialization
{

template <> struct Initializer<Method::Xavier>
{
  public:
	static void initialize(arma::mat &weights, u32 input_units,
						   u32 output_units)
	{
		const auto variance = 2.0 / (input_units + output_units);

		// resize
		weights.resize(input_units, output_units);
		weights.imbue([&]() { return arma::randn() * std::sqrt(variance); });
	}

	static void initialize(arma::vec &weights, u32 input_units)
	{
		const auto variance = 2.0 / (input_units);
		// resize
		weights.resize(input_units);
		weights.imbue([&]() { return arma::randn() * std::sqrt(variance); });
	}
};

}  // namespace Core::Initialization