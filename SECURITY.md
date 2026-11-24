# Security

## Defaults

- The server binds to `127.0.0.1` unless `BIND_HOST` is changed.
- `scripts/serve.sh` refuses a non-loopback bind without an API key. Set
  `API_KEY_FILE` (recommended) or `VLLM_API_KEY`; `ALLOW_INSECURE=1` exists only
  for trusted, isolated networks.
- `.env` is git-ignored. Keep it and the key file at mode `600`.

## API keys

```bash
printf 'your-key\n' > ~/.vllm-key
chmod 600 ~/.vllm-key
# .env
API_KEY_FILE="$HOME/.vllm-key"
```

Do not put keys in `services/*.in`, shell history, benchmark commands or
issues. The benchmark scripts accept `--api-key-file` and also read
`VLLM_API_KEY` from the environment.

## Network exposure

This repository deliberately contains no tunnel, FRP, reverse-proxy or
public-endpoint configuration. For LAN or internet access:

1. Terminate TLS at a reverse proxy.
2. Authenticate at the proxy, not only at vLLM.
3. Restrict the listen port with a firewall/security group.
4. Rotate the API key if it was ever pasted into a chat or log.

## Model and data

- Model weights are not distributed here. Obtain them from the publisher and
  comply with the model license.
- Prompts and completions transit the vLLM API in plain HTTP on loopback. If you
  expose the API, use TLS.

## Reporting

Open an issue without secrets, private IPs or full logs. `scripts/doctor.sh`
prints environment details; review it before sharing.
