use std::collections::HashMap;
use std::net::SocketAddr;
use std::process;
use std::str::FromStr;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use futures::{StreamExt};
use kaspa_addresses::Address;
use kaspa_consensus_core;
use kaspa_consensus_core::Hash;
use kaspa_consensus_core::hashing::header::hash_override_nonce_time;
use kaspa_grpc_client::GrpcClient;
use kaspa_notify::connection::{ChannelConnection, ChannelType};
use kaspa_notify::scope::{NewBlockTemplateScope, Scope};
use kaspa_notify::subscriber::SubscriptionManager;
use kaspa_notify::subscription::context::SubscriptionContext;
use kaspa_rpc_core::{api::rpc::RpcApi, GetBlockTemplateResponse, Notification, notify::mode::NotificationMode};
use log::{debug, error, info, warn};
use serde::{Deserialize, Serialize};
use serde_json::{from_str, to_string};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tonic::codegen::tokio_stream::StreamMap;

use crate::json_rpc;
use crate::cli::Cli;
use crate::state::{Job, JobId, State};
use crate::stratum_server::StratumServer;

pub struct Stratum
{
    cli: Cli,
    client: Option<GrpcClient>,
    clients: Arc<Mutex<StreamMap<SocketAddr, mpsc::Sender<Job>>>>,
    state: Arc<Mutex<State>>,
}

impl Stratum
{
    pub fn new(cli: Cli) -> Self
    {
        let clients = Arc::new(Mutex::new(StreamMap::new()));

        Stratum
        {
            cli,
            client: None,
            clients: clients.clone(),
            state: Arc::new(Mutex::new(State::new(clients.clone(), None))),
        }
    }

    pub async fn init(&mut self) -> Result<()>
    {
        // let address = "172.201.113.73:13110"; // TODO:
        // let address = "127.0.0.1:13110"; // TODO:
        let address = "127.0.0.1:16210"; // TODO:
        self.client = Some(GrpcClient::connect_with_args(
            NotificationMode::MultiListeners,
            format!("grpc://{}", address),
            Some(SubscriptionContext::with_options(None)),
            true,
            None,
            false,
            Some(500_000),
            Default::default(),
        ).await?);
        let client = self.client.clone().unwrap();

        if client.is_connected()
        {
            info!("Connected to node {}", address);
            // TODO: Wait for the node sync
        }
        else
        {
            error!("Failed to connect to node {}", address);
            // TODO: Wait for the node
        }

        let (sender, event_receiver) = async_channel::unbounded();
        let connection = ChannelConnection::new("stratum", sender.clone(), ChannelType::Closable);
        let listener_id = client.register_new_listener(connection);

        let notifier = client.clone().notifier().unwrap();
        notifier.clone().start();

        let clients = self.clients.clone();
        let mut state = self.state.clone();
        let client_clone = client.clone();

        state.clone().lock().unwrap().

        tokio::spawn(async move
        {
            match notifier.start_notify(listener_id, Scope::NewBlockTemplate(NewBlockTemplateScope::default())).await
            {
                Ok(()) =>
                {
                    info!("Starting to listening for new blocks templates ..");
                }

                Err(error) =>
                {
                    error!("Failed to register for new block template: {}", error);
                }
            }

            let mut last_block_hash = Hash::default();

            while let Ok(notification) = event_receiver.recv().await
            {
                match notification
                {
                    Notification::NewBlockTemplate(_) =>
                    {
                        if let Some((block_hash, template)) = Self::on_block_template(&client_clone).await
                        {
                            // Sometimes the template is the same we already had
                            if block_hash == last_block_hash
                            {
                                debug!("Skipped duplicated block of {}", block_hash);
                                return;
                            }

                            last_block_hash = block_hash;

                            state.lock().unwrap().sync(&template.block.header, block_hash).unwrap();
                        }
                    }

                    _ => {}
                }
            }
        });

        // Run the first template block once stratum starts
        if let Some((block_hash, template)) = Self::on_block_template(&client).await
        {
            println!("block_hash: {}", block_hash);
            self.state.lock().unwrap().sync(&template.block.header, block_hash).unwrap();
        }

        Ok(())
    }

    pub async fn start(&self) -> Result<()>
    {
        // TODO: Add --local since default will be 0.0.0.0
        let servers = vec!
        [
            StratumServer::new("0.0.0.0".to_string(), 1500, true, false),
            // StratumServer::new("127.0.0.1".to_string(), 1500, true, false),
        ];

        let clients_clone = self.clients.clone();
        let state_clone = self.state.clone();

        for server in servers
        {
            let clients_clone = clients_clone.clone();
            let state_clone = state_clone.clone();

            tokio::spawn(async move
            {
                let mut socket_stream = server.start().await.unwrap();

                tokio::pin!(socket_stream);

                while let Some(socket_result) = socket_stream.next().await
                {
                    match socket_result
                    {
                        Ok(socket) =>
                        {
                            let addr = socket.peer_addr().unwrap();
                            let (client_tx, mut client_rx) = mpsc::channel::<Job>(1024);

                            clients_clone.lock().unwrap().insert(addr, client_tx);

                            println!("New client connected: {}", addr);
                            println!("Total clients: {}", clients_clone.lock().unwrap().len());

                            let clients_clone = clients_clone.clone();
                            let state_clone = state_clone.clone();

                            tokio::spawn(async move
                            {
                                let _ = Stratum::handle_client(socket, addr, client_rx, state_clone).await.unwrap();

                                clients_clone.lock().unwrap().remove(&addr);

                                println!("Client disconnected: {}", addr);
                                println!("Total clients: {}", clients_clone.lock().unwrap().len());
                            });
                        }
                        Err(e) =>
                        {
                            eprintln!("Error accepting connection: {:?}", e);
                        }
                    }
                }
            });
        }

        Ok(())
    }

    async fn on_block_template(client: &GrpcClient) -> Option<(Hash, GetBlockTemplateResponse)>
    {
        let address = "pyrin:qzn54t6vpasykvudztupcpwn2gelxf8y9p84szksr73me39mzf69uaalnymtx".to_string();
        let pay_address = Address::try_from(address.clone());

        if pay_address.is_err()
        {
            panic!("Failed to parse pay address of {}", address);
        }

        let pay_address = pay_address.unwrap();

        // TODO:
        let template = client.get_block_template(pay_address, vec![0x55]).await;

        if template.is_err()
        {
            error!("Failed to read block template from node: {}", template.err().unwrap());
            return None;
        }

        let template = template.unwrap();

        if !template.is_synced
        {
            warn!("Block template received but node is not synced");
            return None;
        }

        return Some((hash_override_nonce_time(&template.block.header, 0, 0), template));
    }

    async fn handle_client(mut stream: TcpStream, peer_addr: SocketAddr, mut rx: mpsc::Receiver<Job>, state: Arc<Mutex<State>>) -> Result<()>
    {
        let mut buffer = [0; 1024];
        let (mut read_stream, mut write_stream) = stream.into_split();

        let (tx, mut rx_write) = mpsc::channel::<String>(100);

        let job_task = tokio::spawn(async move
        {
            let mut id = 0;

            loop
            {
                tokio::select!
                {
                    Some(job) = rx.recv() =>
                    {
                        let job_id = job.0[3].to_string();

                        if let Ok(message) = json_rpc::notify(job_id, job.0, job.1).await
                        {
                            let message = message + "\n";
                            let _ = write_stream.write_all(message.as_bytes()).await;
                            id += 1;
                        }
                    }
                    Some(msg) = rx_write.recv() =>
                    {
                        let msg = msg + "\n";
                        let _ = write_stream.write_all(msg.as_bytes()).await;
                    }
                    else => break,
                }
            }
        });

        let state_clone = state.clone();

        let client_task = tokio::spawn(async move
        {
            // if let Ok(difficulty_msg) = json_rpc::set_difficulty(32.0).await
            if let Ok(difficulty_msg) = json_rpc::set_difficulty(4.0).await
            {
                let _ = tx.send(difficulty_msg).await;
            }

            loop
            {
                match read_stream.read(&mut buffer).await
                {
                    Ok(0) => break, // Connection closed
                    Ok(n) =>
                        {
                        info!("Received {:?} from client {}", buffer.len(), peer_addr);

                        let received = String::from_utf8_lossy(&buffer[..n]);
                        for json_str in received.split_inclusive("\n")
                        {
                            if let Ok(request) = from_str::<JsonRpcRequest>(json_str.trim())
                            {
                                let response = handle_request(peer_addr, request, state_clone.clone()).await;

                                if let Some(response) = response
                                {
                                    println!("[response]: {:?}", response);
                                    if let Ok(response_str) = to_string(&response)
                                    {
                                        let _ = tx.send(response_str).await;
                                    }
                                }
                            }
                            else
                            {
                                error!("Failed to parse request from miner: {:?}", json_str);
                            }
                        }
                    }
                    Err(e) =>
                    {
                        error!("Error reading from client {}: {}", peer_addr, e);
                        break;
                    }
                }
            }

            info!("Client {} connection closed", peer_addr);

            job_task.abort();
        });

        client_task.await?;

        Ok(())
    }
}

#[derive(Serialize, Deserialize, Debug)]
struct JsonRpcRequest
{
    id: u64,
    method: String,
    params: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug)]
struct JsonRpcResponse<T>
{
    id: u64,
    result: T,
    error: Option<String>,
}

struct ClientState
{
    extranonce: String,
    authorized: bool,
}

struct ServerState
{
    clients: HashMap<String, ClientState>,
    current_job: String,
}

async fn handle_request(peer_addr: SocketAddr, request: JsonRpcRequest, state: Arc<Mutex<State>>) -> Option<JsonRpcResponse<serde_json::Value>>
{
    info!("[Request] ({}) {:?}", peer_addr, request);

    match request.method.as_str()
    {
        "mining.subscribe" =>
        {
            let extranonce = process::id().to_string();
            // let mut state = state.lock().unwrap();
            //
            // state.clients.insert(extranonce.clone(), ClientState
            // {
            //     extranonce: extranonce.clone(),
            //     authorized: false,
            // });

            if let Some(miner) = request.params.get(0)
            {
                let protocol_version = request.params.get(1);
                println!("[miner]: {:?}", miner);
                println!("[protocol_version]: {:?}", protocol_version);

                // json_rpc::notify("1", )
                None
            }
            else
            {
                Some(
                    JsonRpcResponse
                    {
                        id: request.id,
                        result: serde_json::json!(false),
                        error: Some("Invalid parameters".to_string()),
                    }
                )
            }
        }

        "mining.authorize" =>
        {
            // let mut state = state.lock().unwrap();
            if let Some(username) = request.params.get(0)
            {
                println!("[]username]: {}", username);

                // if let Some(client) = state.clients.get_mut(username)
                // {
                //     client.authorized = true;
                //     JsonRpcResponse
                //     {
                //         id: request.id,
                //         result: serde_json::json!(true),
                //         error: None,
                //     }
                // } else {
                //     JsonRpcResponse {
                //         id: request.id,
                //         result: serde_json::json!(false),
                //         error: Some("Client not found".to_string()),
                //     }
                // }
                Some(
                    JsonRpcResponse
                    {
                        id: request.id,
                        result: serde_json::json!(true),
                        error: None,
                    }
                )
            }
            else
            {
                Some(
                    JsonRpcResponse
                    {
                        id: request.id,
                        result: serde_json::json!(false),
                        error: Some("Invalid parameters".to_string()),
                    }
                )
            }
        }

        "mining.submit" =>
        {
            // TODO: We will need to add extranonce probably


            // let state = state.lock().unwrap();
            if request.params.len() >= 3
            {
                let username = &request.params[0];
                let job_id = &request.params[1];
                let nonce = &request.params[2];

                // In a real implementation, verify the submitted work here
                println!("Work submitted by {}: job_id={}, nonce={}", username, job_id, nonce);

                if let Ok(nonce) = State::parse_nonce(nonce.as_str())
                {
                    let pow_valid = state.clone().lock().unwrap().check_pow(JobId::from_str(job_id.as_str()).unwrap(), nonce);

                    // TODO: If valid is block found
                    println!("pow_valid: {} {}", pow_valid, nonce);

                    if pow_valid
                    {

                    }

                    // TODO: Validate the share

                    // Some(
                    //     JsonRpcResponse
                    //     {
                    //         id: request.id,
                    //         result: serde_json::json!(pow_valid),
                    //         error: None,
                    //     }
                    // )

                    Some(
                        JsonRpcResponse
                        {
                            id: request.id,
                            result: serde_json::json!(true),
                            error: None,
                        }
                    )
                }
                else
                {
                    println!("pow_valid: {}", nonce);

                    Some(
                        JsonRpcResponse
                        {
                            id: request.id,
                            result: serde_json::json!(false),
                            error: Some("Invalid nonce".to_string()),
                        }
                    )
                }
            }
            else
            {
                Some(
                    JsonRpcResponse
                    {
                        id: request.id,
                        result: serde_json::json!(false),
                        error: Some("Invalid parameters".to_string()),
                    }
                )
            }
        }
        _ =>
        {
            Some(
                JsonRpcResponse
                {
                    id: request.id,
                    result: serde_json::json!(null),
                    error: Some("Unknown method".to_string()),
                }
            )
        },
    }
}