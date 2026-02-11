use anyhow::Result;
use serde_json::Value;
use std::fs;

pub struct Config;

impl Config {
    pub fn load() -> Result<Value> {
        let content = fs::read_to_string("config.json")?;
        Ok(serde_json::from_str(&content)?)
    }
}
