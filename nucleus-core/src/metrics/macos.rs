use crate::metrics::collector::MetricsCollector;
use crate::metrics::types::ResourceUsage;

pub struct MacOSCollector {
    // TODO: Add system handle for metrics collection
}

impl MacOSCollector {
    pub fn new() -> anyhow::Result<Self> {
        Ok(Self {})
    }
}

impl MetricsCollector for MacOSCollector {
    fn collect(&self) -> anyhow::Result<ResourceUsage> {
        // TODO: Implement actual collection using sysinfo or system APIs
        Ok(ResourceUsage {
            cpu_percent: 0.0,
            gpu_percent: None,
        })
    }
}
