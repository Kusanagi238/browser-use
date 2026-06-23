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

		# Avoid calling model_copy (may not exist on BaseMessage). Just iterate a copy of the list.
		messages = list(messages)

		formatted_messages = []
		system_message: str | None = None

		for message in messages:
			# Use getattr to avoid static attribute access errors on BaseMessage
			role = getattr(message, 'role', None)

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

			# Initialize message parts
			message_parts: list[Part] = []

			# Extract content and create parts
			if isinstance(message.content, str):
				# Regular text content
				message_parts = [Part.from_text(text=message.content)]
			elif message.content is not None:
				# Handle Iterable of content parts
				for part in message.content:
					if getattr(part, 'type', None) == 'text':
						message_parts.append(Part.from_text(text=getattr(part, 'text', '')))
					elif getattr(part, 'type', None) == 'refusal':
						message_parts.append(Part.from_text(text=f'[Refusal] {getattr(part, "refusal", "")}'))
					elif getattr(part, 'type', None) == 'image_url':
						# Handle images robustly: support data-URIs, bytes, or file-like objects
						image_source = getattr(part, 'image_url', None)
						if image_source is None:
							continue
						url = getattr(image_source, 'url', image_source)

						image_bytes = None
						if isinstance(url, str):
							# Expect data URI like: data:image/png;base64,<data>
							if ',' in url:
								try:
									header, data = url.split(',', 1)
									image_bytes = base64.b64decode(data)
								except Exception:
									# Not a valid data URI, skip
									continue
							else:
								# Not a data URI string, skip
								continue
						elif isinstance(url, (bytes, bytearray)):
							image_bytes = bytes(url)
						elif hasattr(url, 'read'):
							try:
								image_bytes = url.read()
							except Exception:
								continue
						else:
							# Unknown image representation; skip
							continue

						# If we have image bytes, create an image part
						if image_bytes is not None:
							image_part = Part.from_bytes(data=image_bytes, mime_type='image/png')
							message_parts.append(image_part)

			# Create the Content object
			if message_parts:
				final_message = Content(role=role, parts=message_parts)
				formatted_messages.append(final_message)

		return formatted_messages, system_message
