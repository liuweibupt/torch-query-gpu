# CoddSpeed Microsoft Research source note

Source: <https://www.microsoft.com/en-us/research/publication/coddspeed-hardware-accelerated-query-processing-in-microsoft-fabric/>

Access date: 2026-06-09.

The page identifies the work as:

- **Title:** CoddSpeed: Hardware Accelerated Query Processing in Microsoft Fabric
- **Venue:** SIGMOD 2026 Industrial Track
- **Date:** May 2026
- **DOI:** <https://doi.org/10.1145/3788853.3803077>

Local download attempt result: no reachable public PDF was found from the MSR
page, and ACM PDF endpoints returned HTTP 403/Cloudflare in this environment.
The repository therefore records the source page and blocker rather than
committing an empty or fake PDF.

Design-relevant public-abstract notes:

- Hardware-independent accelerated analytics in Microsoft Fabric.
- GPU-based execution engine derived from Tensor Query Processor (TQP).
- Variety of accelerators and networks: GPUs, FPGAs, ASICs, NVLink,
  InfiniBand.
- Data movement is a first-class system concern for accelerated analytics.
