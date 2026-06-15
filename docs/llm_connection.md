# LLM connection troubleshooting

The LLM client uses the OpenAI-compatible API configured by `LLM_BASE_URL`,
`LLM_API_KEY`, and `LLM_MODEL`.

If the page shows an LLM call failure with `Connection error.`, first check proxy-related
environment variables such as `HTTP_PROXY`, `HTTPS_PROXY`, and `ALL_PROXY`.
The OpenAI Python SDK reads these variables automatically through `httpx`.

This project now removes the known dead local proxy value `127.0.0.1:9` before
creating the LLM client. That value points requests to a local port where no
proxy service is expected to be running, which causes `APIConnectionError:
Connection error.` before the request reaches the model provider.
