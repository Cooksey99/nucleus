mod types;
mod collector;
mod aggregator;

#[cfg(target_os = "macos")]
mod macos;

pub use types::{
    MetricsSnapshot, PerformanceMetrics, ResourceUsage, MetricsConfig
};
pub use collector::MetricsCollector;
pub use aggregator::MetricsAggregator;

#[cfg(target_os = "macos")]
pub use macos::MacOSCollector;
