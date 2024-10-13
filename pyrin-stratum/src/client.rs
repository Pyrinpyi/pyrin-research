/*use anyhow::Result;
use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;
use crate::json_rpc;

pub async fn on_client_connected(mut stream: &mut TcpStream) -> Result<()>
{
    let _ = stream.write_all(json_rpc::set_difficulty(4).await?.as_bytes()).await;

    // let _ = stream.write_all(serde_json::to_vec(&request)?.as_slice()).await;

    Ok(())
}

pub async fn on_client_disconnected(mut stream: &TcpStream) -> Result<()>
{
    Ok(())
}

pub async fn on_client_error(mut stream: &TcpStream) -> Result<()>
{
    Ok(())
}*/