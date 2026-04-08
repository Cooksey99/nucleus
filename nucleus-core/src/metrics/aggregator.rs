use crate::metrics::types::{MetricsSnapshot, PerformanceMetrics};

pub struct MetricsAggregator {
    snapshots: Vec<MetricsSnapshot>,
}

impl MetricsAggregator {
    pub fn new() -> Self {
        Self {
            snapshots: Vec::new(),
        }
    }

    pub fn add_snapshot(&mut self, snapshot: MetricsSnapshot) {
        self.snapshots.push(snapshot);
    }

    pub fn finalize(
        self,
        model: String,
        provider: String,
        total_duration_ms: u64,
        tokens_generated: usize,
    ) -> PerformanceMetrics {
        let tokens_per_second = if total_duration_ms > 0 {
            (tokens_generated as f64 / (total_duration_ms as f64 / 1000.0)) as f32
        } else {
            0.0
        };

        let (avg_cpu, max_cpu, avg_gpu, max_gpu) = if self.snapshots.is_empty() {
            (0.0, 0.0, None, None)
        } else {
            let avg_cpu = self.snapshots.iter()
                .map(|s| s.resource_usage.cpu_percent)
                .sum::<f32>() / self.snapshots.len() as f32;

            let max_cpu = self.snapshots.iter()
                .map(|s| s.resource_usage.cpu_percent)
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(0.0);

            let gpu_samples: Vec<f32> = self.snapshots.iter()
                .filter_map(|s| s.resource_usage.gpu_percent)
                .collect();

            let (avg_gpu, max_gpu) = if gpu_samples.is_empty() {
                (None, None)
            } else {
                let avg = gpu_samples.iter().sum::<f32>() / gpu_samples.len() as f32;
                let max = gpu_samples.iter()
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .copied()
                    .unwrap_or(0.0);
                (Some(avg), Some(max))
            };

            (avg_cpu, max_cpu, avg_gpu, max_gpu)
        };

        PerformanceMetrics {
            model,
            provider,
            tokens_per_second,
            avg_cpu_percent: avg_cpu,
            max_cpu_percent: max_cpu,
            avg_gpu_percent: avg_gpu,
            max_gpu_percent: max_gpu,
        }
    }
}

impl Default for MetricsAggregator {
    fn default() -> Self {
        Self::new()
    }
}
