use anyhow::Result;
use serde::{Deserialize, Serialize};
use crate::state::JobId;

#[derive(Serialize, Deserialize, Debug)]
pub struct JsonRpcRequest<T>
{
    id: Option<String>,
    method: String,
    params: T,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DifficultyParams([f64; 1]);

#[derive(Serialize, Deserialize, Debug)]
pub struct NotifyParams(JobId, [u64; 4], u64);

#[derive(Serialize, Deserialize, Debug)]
struct SubscribeParams(String, String);

#[derive(Serialize, Deserialize, Debug)]
struct AuthorizeParams(String, String);

#[derive(Serialize, Deserialize, Debug)]
struct SubmitParams(String, String, String);


impl JsonRpcRequest<DifficultyParams>
{
    fn new_set_difficulty(difficulty: f64) -> Self
    {
        Self
        {
            id: None,
            method: "mining.set_difficulty".to_string(),
            params: DifficultyParams([difficulty]),
        }
    }
}


impl JsonRpcRequest<NotifyParams>
{
    fn notify(job_id: JobId, job: [u64; 4], timestamp: u64) -> Self
    {
        Self
        {
            id: None,
            method: "mining.notify".to_string(),
            params: NotifyParams(job_id, job, timestamp),
        }
    }
}

impl JsonRpcRequest<SubscribeParams>
{
    fn new_subscribe(id: String, miner_name: &str, protocol_version: &str) -> Self
    {
        Self
        {
            id: Some(id),
            method: "mining.subscribe".to_string(),
            params: SubscribeParams(miner_name.to_string(), protocol_version.to_string()),
        }
    }
}

impl JsonRpcRequest<AuthorizeParams>
{
    fn new_authorize(id: String, username: &str, password: &str) -> Self
    {
        Self
        {
            id: Some(id),
            method: "mining.authorize".to_string(),
            params: AuthorizeParams(username.to_string(), password.to_string()),
        }
    }
}

impl JsonRpcRequest<SubmitParams>
{
    fn new_submit(id: String, username: &str, job_id: &str, miner_nonce: &str) -> Self
    {
        Self
        {
            id: Some(id),
            method: "mining.submit".to_string(),
            params: SubmitParams(username.to_string(), job_id.to_string(), miner_nonce.to_string()),
        }
    }
}

pub async fn set_difficulty(difficulty: f64) -> Result<String>
{
    return Ok(serde_json::to_string(&JsonRpcRequest::new_set_difficulty(difficulty))?);
}

pub async fn notify(job_id: JobId, job: [u64; 4], timestamp: u64) -> Result<String>
{
    return Ok(serde_json::to_string(&JsonRpcRequest::notify(job_id, job, timestamp))?);
}