use std::collections::{HashMap, HashSet, VecDeque};
use std::io::Cursor;
use anyhow::Result;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use byteorder::{LittleEndian, ReadBytesExt};
use kaspa_consensus_core::Hash;
use kaspa_consensus_core::header::Header;
use kaspa_grpc_client::GrpcClient;
use tokio::sync::mpsc;
use tonic::codegen::tokio_stream::StreamMap;

pub type JobId = String;

#[derive(Clone, Debug)]
pub struct Job(pub(crate) [u64; 4], pub(crate) u64);

pub struct State
{
    inner: Arc<Mutex<HashMap<JobId, kaspa_pow::State>>>,
    job_ids: Arc<Mutex<VecDeque<JobId>>>,
    clients: Arc<Mutex<StreamMap<SocketAddr, mpsc::Sender<Job>>>>,
    client: Option<Arc<GrpcClient>>,
}

impl State
{
    pub fn new(clients: Arc<Mutex<StreamMap<SocketAddr, mpsc::Sender<Job>>>>, client: Option<Arc<GrpcClient>>) -> Self
    {
        State
        {
            inner: Arc::new(Mutex::new(HashMap::<JobId, kaspa_pow::State>::new())),
            job_ids: Arc::new(Mutex::new(VecDeque::<JobId>::new())),
            clients,
            client,
        }
    }

    pub fn sync(&mut self, header: &Header, block_hash: Hash) -> Result<()>
    {
        // kaspa_pow::State::new(header)
        let job = Self::generate_job_header(block_hash.to_string());
        let job_message = Job(job, header.timestamp);
        let job_id = job[3].to_string();

        let inner = self.inner.clone();
        let clients = self.clients.clone();

        inner.lock().unwrap().insert(job_id.clone(), kaspa_pow::State::new(header));

        for client in clients.lock().unwrap().values_mut()
        {
            println!("[job_message]: {:?}", job_message);
            client.try_send(job_message.clone())?;
        }

        let mut job_ids = self.job_ids.lock().unwrap();
        job_ids.push_back(job_id.clone());

        if job_ids.len() >= 100
        {
            if let Some(first_job_id) = job_ids.pop_front()
            {
                inner.lock().unwrap().remove(&first_job_id);
            }
        }

        Ok(())
    }

    pub fn check_pow(&self, job_id: JobId, nonce: u64) -> bool
    {
        let inner = self.inner.clone();
        let inner = inner.lock().unwrap();
        let state = inner.get(&job_id).unwrap();

        let (valid, _) = state.check_pow(nonce);

        return valid;
    }

    fn generate_job_header(hash: String) -> [u64; 4]
    {
        let header_data = hex::decode(hash).expect("Invalid hex string");
        let mut ids = Vec::with_capacity(4);
        let mut rdr = Cursor::new(header_data);

        for _ in 0..4
        {
            ids.push(rdr.read_u64::<LittleEndian>().unwrap());
        }

        ids.iter()
            .map(|&v|
            {
                let as_hex = format!("{:x}", v);
                u64::from_str_radix(&as_hex, 16).unwrap()
            })
            .collect::<Vec<u64>>()
            .try_into()
            .unwrap()
    }

    pub fn parse_nonce(nonce: &str) -> Result<u64, std::num::ParseIntError>
    {
        u64::from_str_radix(nonce.trim_start_matches("0x"), 16)
    }
}

#[cfg(test)]
mod tests
{
    use std::str::FromStr;
    use super::*;

    #[test]
    fn test_generate_job_header()
    {
        assert_eq!(
            State::generate_job_header("634523821e7a7094b7c9c4ca02060ccd074259d70209880236188a74b2946a70".to_string()),
            [10696183386455885155, 14775191086557350327, 182405692716040711, 8100450373959555126]
        );
    }

    #[test]
    fn test_parse_nonce()
    {
        assert_eq!(State::parse_nonce("0x006ac2fab807209a").unwrap(), 30050729616416922u64)
    }
}