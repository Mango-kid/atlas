// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;
use spark_runtime::gpu::{GpuBackend, GraphHandle};
use std::{collections::VecDeque, sync::Arc};

/// Bounded LRU of prefix shapes. Graphs reference this model's resident buffers;
/// observations and noise are uploaded afresh before every replay.
pub(super) struct GraphCache {
    gpu: Arc<dyn GpuBackend>,
    entries: VecDeque<(usize, GraphHandle)>,
}

impl GraphCache {
    pub fn new(gpu: Arc<dyn GpuBackend>) -> Self {
        Self {
            gpu,
            entries: VecDeque::new(),
        }
    }

    pub fn get(&mut self, rows: usize) -> Option<GraphHandle> {
        let index = self.entries.iter().position(|(n, _)| *n == rows)?;
        let entry = self.entries.remove(index)?;
        self.entries.push_back(entry);
        Some(entry.1)
    }

    pub fn insert(&mut self, rows: usize, graph: GraphHandle) -> Result<()> {
        const MAX_SHAPES: usize = 8;
        // Record ownership before a fallible eviction.
        self.entries.push_back((rows, graph));
        if self.entries.len() > MAX_SHAPES
            && let Some((_, old)) = self.entries.pop_front()
        {
            self.gpu.destroy_graph(old)?;
        }
        Ok(())
    }
}

impl Drop for GraphCache {
    fn drop(&mut self) {
        let _ = self.gpu.synchronize(self.gpu.default_stream());
        for (_, graph) in self.entries.drain(..) {
            let _ = self.gpu.destroy_graph(graph);
        }
    }
}
