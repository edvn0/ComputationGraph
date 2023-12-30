#pragma once

#include <exception>
#include <string>

namespace Core
{

class BaseException : public std::exception
{
  public:
	BaseException() = default;
	explicit BaseException(const char *message) : message(message)
	{
	}
	explicit BaseException(const std::string &message) : message(message)
	{
	}
	~BaseException() override = default;

	[[nodiscard]] auto what() const noexcept -> const char * override
	{
		return message.c_str();
	}

  private:
	std::string message;
};

class IncompatibleLayerDefinitionException : public BaseException
{
  public:
	using BaseException::BaseException;
	IncompatibleLayerDefinitionException() = default;
	~IncompatibleLayerDefinitionException() override = default;
};

class IncompatibleInputException : public BaseException
{
  public:
	using BaseException::BaseException;
	IncompatibleInputException() = default;
	~IncompatibleInputException() override = default;
};

}  // namespace Core