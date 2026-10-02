# 15 · Data-Acquisition Diagnosis & Source Policy (2026-09-13)

Records the network diagnosis that blocks the independent-cohort data pull, the exact evidence,
and the policy for any substitute source. Produced while the structural-validation framework runs
were in flight; no experiment data or existing dataset was modified by the diagnosis itself.

## 1. Symptom

Every HTTPS request from this host that goes through the Windows proxy at `127.0.0.1:7897` dies
with `_ssl.c:1011: The handshake operation timed out`. The failure affects *all* destinations
(including hosts that had worked earlier), which initially looked like a Yahoo-wide outage.

## 2. What the diagnostics found (recorded verbatim by category)

| Layer | Observation |
|---|---|
| System proxy (registry) | `ProxyEnable=1`, `ProxyServer=127.0.0.1:7897`, override = intranet ranges only |
| WinHTTP proxy | direct (no proxy) |
| Process env vars | `HTTP_PROXY`/`HTTPS_PROXY`/`http_proxy`/`https_proxy` = `http://127.0.0.1:7897`; `NO_PROXY` empty; user/machine level unset |
| Proxy client | `clash-verge` (pid 26508) and `verge-mihomo` (pid 29648) running; port 7897 listening; no other common proxy port listening |
| DNS | query1/query2 resolve to IPv6 first (`2001:4998:…`) plus IPv4 `69.147.80.12` / `69.147.80.15`. Anomaly: `www.google.com` resolves to `2001::1` and `31.13.92.37` (a Facebook address) → suspected DNS interference on one interface. Two NICs use different resolvers (Ethernet → 192.168.31.1; WLAN → 223.5.5.5/223.6.6.6) |
| TLS (direct) | Handshake to `69.147.80.12:443` and `.15:443` with SNI `query1.finance.yahoo.com` **succeeds** (TLSv1.3, `TLS_AES_128_GCM_SHA256`) → TLS layer is healthy |

### AAPL chart-endpoint matrix (`/v8/finance/chart/AAPL`)

| Mode | Result |
|---|---|
| inherit system proxy (query1 / query2) | `URLError: handshake timed out` |
| explicit bypass (`ProxyHandler({})`) | `HTTPError 403` + Yahoo "sad panda" HTML (application-level block) |
| explicit proxy `127.0.0.1:7897` | `URLError: handshake timed out` |
| child process with **all** proxy env vars removed | still times out — urllib on Windows also reads the **registry** proxy |
| control: `www.cloudflare.com` | `HTTP 200` both with and without the proxy → the proxy *can* tunnel TLS, and the network works |
| control: `pypi.tuna.tsinghua.edu.cn` | `403` (normal response on that path) |

## 3. Conclusion

1. The timeouts are caused solely by routing through `127.0.0.1:7897` (env vars **and** registry).
   The Clash client is up, but its current upstream cannot reach Yahoo while it *can* reach
   Cloudflare.
2. With the proxy bypassed, TLS to Yahoo works but Yahoo answers **403** — an **IP/region-level
   application block**, not a proxy, DNS, or TLS defect.
3. Therefore unsetting environment variables alone cannot fix Yahoo access: without them the
   registry proxy still applies (timeout), and with a true bypass the answer is 403.

## 4. Policy and options (none executed without explicit approval)

- **Preferred:** operator selects a working Clash node; the agent then verifies with a
  **single-symbol (AAPL) test** and stops on success. **No batch download without approval.**
- **Fallback (requires approval, must be documented as a provenance deviation):** a paged
  Tencent (`web.ifzq.gtimg.cn`) fetcher. Verified states: `usAAPL.OQ` and `hkHSI` return data
  with the correct exchange suffix, but the API caps responses at ≈640 rows per call, so 2016–2025
  needs backward paging; FX and commodity codes are unverified. Sohu has no US equities; East
  Money answers with `RemoteDisconnected` (anti-bot).
- **No composite cohorts.** If a fallback source is used, the whole validation cohort must come from
  one source, and the provenance deviation must be stated next to any result (the reviewer's
  requirement: a same-source cohort, and explicit disclosure of the provider change).

## 5. Consequence for the research line

The independent-cohort validation cannot start until one of the two paths is taken. This does not
affect tonight's Gate A, which uses only assets already on disk.
