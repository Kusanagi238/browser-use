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

		messages = [m.model_copy(deep=True) for m in messages]

		formatted_messages: ContentListUnion = []
		system_message: str | None = None

		for message in messages:
			role = message.role if hasattr(message, 'role') else None

			# Handle system/developer messages
			if isinstance(message, SystemMessage) or role in ['system', 'developer']:
				# Extract system message content as string
				if isinstance(message.content, str):
					system_message = message.content
				# Ensure content is iterable (but not a str/bytes) before iterating
				elif message.content is not None and hasattr(message.content, '__iter__') and not isinstance(message.content, (str, bytes)):
					# Handle Iterable of content parts
					parts = []
					for part in message.content:
						# Skip elements that don't look like parts
						if not hasattr(part, 'type'):
							continue
						if part.type == 'text':
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

			# Initialize message parts
			message_parts: list[Part] = []

			# Extract content and create parts
			if isinstance(message.content, str):
				# Regular text content
				message_parts = [Part.from_text(text=message.content)]
			# Ensure content is iterable (but not a str/bytes) before iterating
			elif message.content is not None and hasattr(message.content, '__iter__') and not isinstance(message.content, (str, bytes)):
				# Handle Iterable of content parts
				for part in message.content:
					# Skip elements that don't look like parts
					if not hasattr(part, 'type'):
						continue
					if part.type == 'text':
						message_parts.append(Part.from_text(text=getattr(part, 'text', '')))
					elif part.type == 'refusal':
						ref_text = getattr(part, 'refusal', '')
						message_parts.append(Part.from_text(text=f'[Refusal] {ref_text}'))
					elif part.type == 'image_url':
						# Handle images
						image_url_obj = getattr(part, 'image_url', None)
						url = getattr(image_url_obj, 'url', None) if image_url_obj is not None else None
						if not isinstance(url, str):
							continue
						# Format: data:image/png;base64,<data>
						try:
							header, data = url.split(',', 1)
							# Decode base64 to bytes
							image_bytes = base64.b64decode(data)
							# Add image part
							image_part = Part.from_bytes(data=image_bytes, mime_type='image/png')
							message_parts.append(image_part)
						except Exception:
							# Skip malformed image URLs without failing
							continue

			# Create the Content object
			if message_parts:
				final_message = Content(role=role, parts=message_parts)
				formatted_messages.append(final_message)

		return formatted_messages, system_message
