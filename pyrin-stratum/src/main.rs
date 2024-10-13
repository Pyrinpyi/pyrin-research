use anyhow::Result;
use log::info;

mod cli;
mod stratum;
mod stratum_server;
mod json_rpc;
mod client;
mod state;
// TODO: https://explorer.pyrin.network/pool/local (to view local stats)
//  https://explorer.pyrin.network/pool/local/192.168.1.177 (to view of another host)

#[tokio::main]
async fn main() -> Result<()>
{
    let cli_result = cli::setup();

    let mut builder = env_logger::Builder::new();

    if cli_result.debug > 1 { builder.filter_level(log::LevelFilter::Trace); }
    else if cli_result.debug > 0 { builder.filter_level(log::LevelFilter::Debug); }
    else { builder.filter_level(log::LevelFilter::Info); }

    builder.init();

    // Run stratum
    if cli_result.command.is_none()
    {
        let mut s = stratum::Stratum::new(cli_result.clone());

        s.init().await?;
        s.start().await?;

        tokio::signal::ctrl_c().await?;
        info!("Received ctrl-c signal, shutting down...");
    }

    Ok(())
}
