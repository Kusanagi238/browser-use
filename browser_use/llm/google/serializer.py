import base64

from browser_use.llm.messages import (
	AssistantMessage,
	BaseMessage,
	SystemMessage,
	UserMessage,
)


class GoogleMessageSerializer:
	"""Serializer for converting messages to Google Gemini format."""

	@staticmethod
	def serialize_messages(messages: list[BaseMessage]) -> tuple[list[dict], str | None]:
		"""
		Convert a list of BaseMessages to Google format, extracting system message.

		Google handles system instructions separately from the conversation, so we need to:
		1. Extract any system messages and return them separately as a string
		2. Convert the remaining messages to Content objects

		Args:
		    messages: List of messages to convert

		Returns:
		    A tuple of (formatted_messages, system_message) where:
		    - formatted_messages: List of dicts representing content for the conversation
		    - system_message: System instruction string or None
		"""

		messages = [m.model_copy(deep=True) for m in messages]

		formatted_messages: list[dict] = []
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

			# Initialize message parts as simple dict structures to avoid strict google types
			message_parts: list[dict] = []

			# Extract content and create parts
			if isinstance(message.content, str):
				# Regular text content
				message_parts = [{"type": "text", "text": message.content}]
			elif message.content is not None:
				# Handle Iterable of content parts
				for part in message.content:
					ptype = getattr(part, 'type', None)
					if ptype == 'text':
						message_parts.append({"type": "text", "text": getattr(part, 'text', '')})
					elif ptype == 'refusal':
						message_parts.append({"type": "text", "text": f"[Refusal] {getattr(part, 'refusal', '')}"})
					elif ptype == 'image_url':
						# Handle images
						url = getattr(part, 'image_url', None)
						if url is not None:
							# url may be a string like data:image/png;base64,<data>
							if isinstance(url, str):
								header, data = url.split(',', 1)
								image_bytes = base64.b64decode(data)
								message_parts.append({"type": "image_bytes", "data": image_bytes, "mime_type": header.split(';')[0].replace('data:', '')})
							else:
								# If image_url is an object with .url
								u = getattr(url, 'url', None)
								if isinstance(u, str):
									header, data = u.split(',', 1)
									image_bytes = base64.b64decode(data)
									message_parts.append({"type": "image_bytes", "data": image_bytes, "mime_type": header.split(';')[0].replace('data:', '')})

			# Create the content dictionary (avoid constructing google-specific Content objects)
			if message_parts:
				final_message = {"role": role, "parts": message_parts}
				formatted_messages.append(final_message)

		return formatted_messages, system_message
