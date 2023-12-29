#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>

namespace Core
{

template <class T, class V>
	requires std::is_default_constructible_v<V>
class PointerMap
{
  public:
	void map_value(std::string_view key, std::shared_ptr<T> pointer)
	{
		const auto as_string = std::string{key};
		pointers[as_string] = pointer;
		values[as_string] = V();
	}

	template <class Other>
		requires(std::is_base_of_v<Other, T>)
	void map_value(std::string_view key, std::shared_ptr<Other> pointer)
	{
		// Cast the pointer ruthlessly to the correct type
		auto cast_pointer = std::dynamic_pointer_cast<T>(pointer);
		const auto as_string = std::string{key};
		pointers[as_string] = cast_pointer;
		values[as_string] = V();
	}

	auto get(std::string_view key) -> V &
	{
		if (!pointers.contains(key.data()))
		{
			throw std::runtime_error("Key not found in PointerMap");
		}

		return values.at(key.data());
	}

	auto get(std::string_view key) const -> const V &
	{
		if (!pointers.contains(key.data()))
		{
			throw std::runtime_error("Key not found in PointerMap");
		}

		return values.at(key.data());
	}

	auto operator[](std::string_view name) -> decltype(auto)
	{
		auto &value = get(name);
		const auto as_string = std::string(name);
		if (pointers.contains(name.data()))
		{
			pointers[as_string]->setValue(value);
		}
		return value;
	}

  private:
	std::unordered_map<std::string, V> values;
	std::unordered_map<std::string, std::shared_ptr<T>> pointers;
};

}  // namespace Core