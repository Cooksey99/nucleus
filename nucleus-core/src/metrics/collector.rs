use crate::metrics::types::{MetricsSnapshot, ResourceUsage};
use std::time::Instant;

pub trait MetricsCollector: Send + Sync {
    fn collect(&self) -> anyhow::Result<ResourceUsage>;
    
    fn create_snapshot(&self, start: Instant) -> anyhow::Result<MetricsSnapshot> {
        Ok(MetricsSnapshot {
            timestamp: start.elapsed(),
            resource_usage: self.collect()?,
        })
    }
}
