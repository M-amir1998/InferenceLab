"""OpenAI-compatible wire format: request schemas, response builders and SSE parsing.

Shared by the gateway, the HTTP backend adapters, the mock serving app and the
benchmark client so that every hop speaks exactly the same dialect.
"""
