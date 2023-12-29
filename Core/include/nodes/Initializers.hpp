#include <cstdint>
#include <stdexcept>

namespace Core::Initialization
{

enum class Method
{
	Xavier,
};

template <Method Method> struct Initializer
{
	template <class T> static void initialize(T &, std::uint32_t, std::uint32_t)
	{
		throw std::runtime_error("Not implemented");
	}

	template <class T> static void initialize(T &, std::uint32_t)
	{
		throw std::runtime_error("Not implemented");
	}
};

}  // namespace Core::Initialization

#include "nodes/initializers/Xavier.inl"