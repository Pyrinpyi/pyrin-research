


### Pyrin Stratum
- Built with Rust for high performance and security
- Fully compatible with [EthereumStratum/1.0](https://github.com/nicehash/Specifications/blob/master/EthereumStratum_NiceHash_v1.0.0.txt) and [EthereumStratum/2.0.0 (EIP-1571)](https://github.com/ethereum/EIPs/blob/master/EIPS/eip-1571.md)
- Optional Redis integration for global state management and pub/sub
  - Which also includes more advanced state handling e.g. graceful shutdown with ACPI to correctly scaling down and up in the clouds


TODO: 
- extranonce supported ?
- EthereumStratum/2.0 ?
- Graceful shutdown (also even sent by Azure to stop VM etc) SIGTERM SIGKILL SIGINT ACPI