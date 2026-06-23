import base64

from google.genai.types import Content, ContentListUnion, Part

from browser_use.llm.messages import (
	AssistantMessage,
	BaseMessage,
	SystemMessage,
	UserMessage,
)


class GoogleMessageSerializer:
	"""Serializer for converting messages to Google Gemini format."""

	@staticmethod
	def serialize_messages(messages: list[BaseMessage]) -> tuple[ContentListUnion, str | None]:
		"""
		Convert a list of BaseMessages to Google format, extracting system message.

		Google handles system instructions separately from the conversation, so we need to:
		1. Extract any system messages and return them separately as a string
		2. Convert the remaining messages to Content objects

		Args:
		    messages: List of messages to convert

		Returns:
		    A tuple of (formatted_messages, system_message) where:
		    - formatted_messages: List of Content objects for the conversation
		    - system_message: System instruction string or None
		"""

		# Use a safe deepcopy instead of calling model_copy which may not exist on BaseMessage
		import copy
		from typing import Any, cast

		messages = [copy.deepcopy(m) for m in messages]

		# Build a list of dynamic objects (Any) and cast back to ContentListUnion on return
		formatted_messages: list[Any] = []
		system_message: str | None = None

		for message in messages:
			role = message.role if hasattr(message, 'role') else None

			# Handle system/developer messages
			if isinstance(message, SystemMessage) or role in ['system', 'developer']:
				# Extract system message content as string
				if isinstance(message.content, str):
					system_message = message.content
				elif message.content is not None:
					# Handle Iterable of content parts
					parts = []
					for part in message.content:
						if getattr(part, 'type', None) == 'text':
							parts.append(getattr(part, 'text', ''))
					system_message = '\n'.join(parts)
				continue

			# Determine the role for non-system messages
			if isinstance(message, UserMessage):
				role = 'user'
			elif isinstance(message, AssistantMessage):
				role = 'model'
			else:
				# Default to user for any unknown message types
				role = 'user'

			# Initialize message parts (use Any to avoid strict typing on factory return types)
			message_parts: list[Any] = []

			# Helper factories - call them dynamically to avoid static signature checks
			text_factory = getattr(Part, 'from_text', None)
			bytes_factory = getattr(Part, 'from_bytes', None)

			# Extract content and create parts
			if isinstance(message.content, str):
				# Regular text content
				if callable(text_factory):
					try:
						message_parts = [text_factory(message.content)]
					except TypeError:
						# fallback positional
						message_parts = [text_factory(message.content)]
				else:
					message_parts = [{'type': 'text', 'text': message.content}]
			elif message.content is not None:
				# Handle Iterable of content parts
				for part in message.content:
					ptype = getattr(part, 'type', None)
					if ptype == 'text':
						text = getattr(part, 'text', '')
						if callable(text_factory):
							try:
								message_parts.append(text_factory(text))
							except TypeError:
								message_parts.append(text_factory(text))
						else:
							message_parts.append({'type': 'text', 'text': text})
					elif ptype == 'refusal':
						ref = getattr(part, 'refusal', '')
						if callable(text_factory):
							message_parts.append(text_factory(f'[Refusal] {ref}'))
						else:
							message_parts.append({'type': 'text', 'text': f'[Refusal] {ref}'})
					elif ptype == 'image_url':
						# Handle images
						url = getattr(part, 'image_url', None)
						url = getattr(url, 'url', url) if url is not None else None

						# Format: data:image/png;base64,<data>
						if isinstance(url, str) and ',' in url:
							header, data = url.split(',', 1)
							# Decode base64 to bytes
							image_bytes = base64.b64decode(data)

							# Add image part using dynamic factory if available
							if callable(bytes_factory):
								# Try common argument patterns
								try:
									image_part = bytes_factory(data=image_bytes, mime_type='image/png')
								except TypeError:
									try:
										image_part = bytes_factory(image_bytes, 'image/png')
									except TypeError:
										image_part = bytes_factory(image_bytes)
							else:
								image_part = {'type': 'image', 'data': image_bytes, 'mime_type': 'image/png'}

							message_parts.append(image_part)

			# Create the Content object or fallback to a dict representation
			if message_parts:
				ContentAny = cast(Any, Content)
				create_fn = getattr(ContentAny, 'create', None)
				if callable(create_fn):
					try:
						final_message = create_fn(role=role, parts=message_parts)
					except TypeError:
						# Try alternate kw name
						try:
							final_message = create_fn(role=role, content=message_parts)
						except Exception:
							final_message = {'role': role, 'parts': message_parts}
				else:
					# Try direct construction dynamically; if it fails, fallback to dict
					try:
						final_message = ContentAny(role=role, parts=message_parts)
					except Exception:
						final_message = {'role': role, 'parts': message_parts}

				formatted_messages.append(final_message)

		# Cast back to the declared return type to satisfy callers and static checkers
		return cast(ContentListUnion, formatted_messages), system_message
