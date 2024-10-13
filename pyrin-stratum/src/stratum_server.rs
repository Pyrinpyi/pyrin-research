use std::pin::Pin;

use anyhow::Result;
use futures::stream::Stream;
use kaspa_consensus_core;
use log::info;
use tokio::net::{TcpListener, TcpStream};

pub struct StratumServer
{
    port: u16,
    address: String,
    solo: bool,
    tls: bool,
}

impl StratumServer
{
    pub fn new(address: String, port: u16, solo: bool, tls: bool) -> Self
    {
        StratumServer
        {
            address,
            port,
            solo,
            tls,
        }
    }

    pub async fn start(&self) -> Result<impl Stream<Item = Result<TcpStream>> + '_>
    {
        let address = format!("{}:{}", self.address, self.port);
        let listener = TcpListener::bind(&address).await?;
        info!("Server listening on {} (solo={}, tls={})", address, self.solo, self.tls);

        let stream = async_stream::try_stream!
        {
            loop
            {
                let (socket, addr) = listener.accept().await?;
                info!("Client {} connected", addr);

                yield socket;
            }
        };

        Ok(stream)
    }
}