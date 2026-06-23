import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeVar, overload

import httpx
from anthropic import (
	NOT_GIVEN,
	APIConnectionError,
	APIStatusError,
	AsyncAnthropic,
	NotGiven,
	RateLimitError,
)
from anthropic.types import CacheControlEphemeralParam, Message, ToolParam
from anthropic.types.model_param import ModelParam
from anthropic.types.tool_choice_tool_param import ToolChoiceToolParam
from httpx import Timeout
from pydantic import BaseModel

from browser_use.llm.anthropic.serializer import AnthropicMessageSerializer
from browser_use.llm.base import BaseChatModel
from browser_use.llm.exceptions import ModelProviderError, ModelRateLimitError
from browser_use.llm.messages import BaseMessage
from browser_use.llm.schema import SchemaOptimizer
from browser_use.llm.views import ChatInvokeCompletion, ChatInvokeUsage

T = TypeVar('T', bound=BaseModel)


@dataclass
class ChatAnthropic(BaseChatModel):
	"""
	A wrapper around Anthropic's chat model.
	"""

	# Model configuration
	model: str | ModelParam
	max_tokens: int = 8192
	temperature: float | None = None

	# Client initialization parameters
	api_key: str | None = None
	auth_token: str | None = None
	base_url: str | httpx.URL | None = None
	timeout: float | Timeout | None | NotGiven = NotGiven()
	max_retries: int = 10
	default_headers: Mapping[str, str] | None = None
	default_query: Mapping[str, object] | None = None

	# Static
	@property
	def provider(self) -> str:
		return 'anthropic'

	def _get_client_params(self) -> dict[str, Any]:
		"""Prepare client parameters dictionary."""
		# Define base client params
		base_params = {
			'api_key': self.api_key,
			'auth_token': self.auth_token,
			'base_url': self.base_url,
			'timeout': self.timeout,
			'max_retries': self.max_retries,
			'default_headers': self.default_headers,
			'default_query': self.default_query,
		}

		# Create client_params dict with non-None values and skip the module-level NOT_GIVEN sentinel
		client_params = {}
		for k, v in base_params.items():
			# Compare to the module-level sentinel NOT_GIVEN rather than constructing a new sentinel
			if v is not None and v is not NOT_GIVEN:
				client_params[k] = v

		return client_params

	def _get_client_params_for_invoke(self):
		"""Prepare client parameters dictionary for invoke."""

		client_params = {}

		if self.temperature is not None:
			client_params['temperature'] = self.temperature

		if self.max_tokens is not None:
			client_params['max_tokens'] = self.max_tokens

		return client_params

	def get_client(self) -> AsyncAnthropic:
		"""
		Returns an AsyncAnthropic client.

		Returns:
			AsyncAnthropic: An instance of the AsyncAnthropic client.
		"""
		client_params = self._get_client_params()
		return AsyncAnthropic(**client_params)

	@property
	def name(self) -> str:
		return str(self.model)

	def _get_usage(self, response: Message) -> ChatInvokeUsage | None:
		usage = ChatInvokeUsage(
			prompt_tokens=response.usage.input_tokens
			+ (
				response.usage.cache_read_input_tokens or 0
			),  # Total tokens in Anthropic are a bit fucked, you have to add cached tokens to the prompt tokens
			completion_tokens=response.usage.output_tokens,
			total_tokens=response.usage.input_tokens + response.usage.output_tokens,
			prompt_cached_tokens=response.usage.cache_read_input_tokens,
			prompt_cache_creation_tokens=response.usage.cache_creation_input_tokens,
			prompt_image_tokens=None,
		)
		return usage

	@overload
	async def ainvoke(self, messages: list[BaseMessage], output_format: None = None) -> ChatInvokeCompletion[str]: ...

	@overload
	async def ainvoke(self, messages: list[BaseMessage], output_format: type[T]) -> ChatInvokeCompletion[T]: ...

	async def ainvoke(
		self, messages: list[BaseMessage], output_format: type[T] | None = None
	) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
		anthropic_messages, system_prompt = AnthropicMessageSerializer.serialize_messages(messages)

		try:
			# Build base params for the create call to avoid passing the NOT_GIVEN sentinel unless needed
			base_create_params: dict[str, Any] = {
				'model': self.model,
				'messages': anthropic_messages,
				**self._get_client_params_for_invoke(),
			}

			# Only include system if a concrete system prompt is present
			if system_prompt is not None and system_prompt is not NOT_GIVEN:
				base_create_params['system'] = system_prompt

			if output_format is None:
				# Normal completion without structured output
				response = await self.get_client().messages.create(**base_create_params)

				usage = self._get_usage(response)

				# Extract text from the first content block in a defensive way
				first_content = None
				if hasattr(response, 'content') and response.content:
					first_content = response.content[0]

				if first_content is None:
					response_text = ''
				elif hasattr(first_content, 'text'):
					response_text = first_content.text
				elif isinstance(first_content, str):
					response_text = first_content
				elif hasattr(first_content, 'content'):
					response_text = str(first_content.content)
				else:
					# Fallback to string representation for unknown block types
					response_text = str(first_content)

				return ChatInvokeCompletion(
					completion=response_text,
					usage=usage,
				)

			else:
				# Use tool calling for structured output
				# Create a tool that represents the output format
				tool_name = output_format.__name__
				schema = SchemaOptimizer.create_optimized_json_schema(output_format)

				# Remove title from schema if present (Anthropic doesn't like it in parameters)
				if 'title' in schema:
					del schema['title']

				tool = ToolParam(
					name=tool_name,
					description=f'Extract information in the format of {tool_name}',
					input_schema=schema,
					cache_control=CacheControlEphemeralParam(type='ephemeral'),
				)

				# Force the model to use this tool
				tool_choice = ToolChoiceToolParam(type='tool', name=tool_name)

				# Add tools and tool_choice to params
				create_params = dict(base_create_params)
				create_params['tools'] = [tool]
				create_params['tool_choice'] = tool_choice

				response = await self.get_client().messages.create(**create_params)

				usage = self._get_usage(response)

				# Extract the tool use block
				for content_block in getattr(response, 'content', []):
					if getattr(content_block, 'type', None) == 'tool_use':
						# Obtain the raw input value from likely attributes in a defensive manner
						input_val = getattr(content_block, 'input', None)
						if input_val is None:
							# Try other common attribute names
							if hasattr(content_block, 'text'):
								input_val = content_block.text
							elif hasattr(content_block, 'content'):
								input_val = content_block.content

						# Normalize and validate the input for the output_format
						try:
							# If it's a string, try JSON first, then pass raw string
							if isinstance(input_val, str):
								try:
									data = json.loads(input_val)
								except Exception:
									data = input_val
							else:
								# If it's not a string, assume it's already a mapping/object
								data = input_val

							validated = output_format.model_validate(data)
							return ChatInvokeCompletion(completion=validated, usage=usage)
						except Exception as e:
							# Propagate validation/parsing errors so callers can handle them
							raise e

				# If no tool use block found, raise an error
				raise ValueError('Expected tool use in response but none found')

		except APIConnectionError as e:
			raise ModelProviderError(message=e.message, model=self.name) from e
		except RateLimitError as e:
			raise ModelRateLimitError(message=e.message, model=self.name) from e
		except APIStatusError as e:
			raise ModelProviderError(message=e.message, status_code=e.status_code, model=self.name) from e
		except Exception as e:
			raise ModelProviderError(message=str(e), model=self.name) from e
