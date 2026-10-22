// SPDX-License-Identifier: AGPL-3.0-only
//! Exact resident rotary tables and combined query/key rotation.
use crate::ops::Ops;
use anyhow::{Result, ensure};
use spark_runtime::{gpu::DevicePtr, kernel_args::KernelLaunch};

impl Ops<'_> {
    /// Table stores BF16 cosine/sine pairs for every position and half-dimension.
    pub fn prepare_rope_table(&self, table: DevicePtr, positions: usize, dim: usize) -> Result<()> {
        ensure!(
            positions > 0 && dim > 0 && dim.is_multiple_of(2),
            "RoPE table geometry"
        );
        let count = u32::try_from(
            positions
                .checked_mul(dim / 2)
                .ok_or_else(|| anyhow::anyhow!("RoPE table size overflow"))?,
        )?;
        KernelLaunch::new(self.gpu, self.gpu.kernel("pi05", "pi05_rope_table")?)
            .grid([count.div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(table)
            .arg_u32(u32::try_from(positions)?)
            .arg_u32(u32::try_from(dim)?)
            .launch(self.stream)
    }

    /// Positions must address the table initialized by `prepare_rope_table`.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_cached(
        &self,
        query: DevicePtr,
        key: Option<DevicePtr>,
        positions: DevicePtr,
        table: DevicePtr,
        rows: usize,
        heads: usize,
        dim: usize,
    ) -> Result<()> {
        ensure!(
            rows > 0 && heads > 0 && dim > 0 && dim.is_multiple_of(2),
            "cached RoPE geometry"
        );
        let count = u32::try_from(
            heads
                .checked_add(usize::from(key.is_some()))
                .and_then(|h| rows.checked_mul(h))
                .and_then(|n| n.checked_mul(dim / 2))
                .ok_or_else(|| anyhow::anyhow!("cached RoPE size overflow"))?,
        )?;
        KernelLaunch::new(self.gpu, self.gpu.kernel("pi05", "pi05_rope_cached")?)
            .grid([count.div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(query)
            .arg_ptr(key.unwrap_or(DevicePtr::NULL))
            .arg_ptr(positions)
            .arg_ptr(table)
            .arg_u32(u32::try_from(rows)?)
            .arg_u32(u32::try_from(heads)?)
            .arg_u32(u32::try_from(dim)?)
            .launch(self.stream)
    }

    /// Rotate an eight-head query and shared one-head key in one launch.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_pair(
        &self,
        query: DevicePtr,
        key: DevicePtr,
        positions: DevicePtr,
        rows: usize,
        heads: usize,
        dim: usize,
    ) -> Result<()> {
        if let Some(table) = self.rope_table {
            self.rope_cached(query, Some(key), positions, table, rows, heads, dim)
        } else {
            self.rope(query, positions, rows, heads, dim)?;
            self.rope(key, positions, rows, 1, dim)
        }
    }
}

impl Ops<'_> {
    /// Rotate Q/current K and clear the complete cache tail before attention.
    /// Both owned caches require `capacity_rows * 256` writable BF16 elements.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_attention_mqa_cached(
        &self,
        query: DevicePtr,
        key: DevicePtr,
        value: DevicePtr,
        positions: DevicePtr,
        out: DevicePtr,
        rows: usize,
        prefix_rows: usize,
        capacity_rows: usize,
        scores: DevicePtr,
    ) -> Result<()> {
        ensure!(rows > 0 && rows <= 712, "unsupported rotary MQA rows");
        let keys = prefix_rows
            .checked_add(rows)
            .ok_or_else(|| anyhow::anyhow!("rotary MQA size overflow"))?;
        let padded = crate::ops_attention::cached_padded_rows(keys, capacity_rows)?;
        if let Some(table) = self.rope_table {
            let count = (rows * 9 * 128).max((padded - keys) * 256);
            KernelLaunch::new(
                self.gpu,
                self.gpu.kernel("pi05", "pi05_rope_cached_clear_tail")?,
            )
            .grid([u32::try_from(count)?.div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(query)
            .arg_ptr(key)
            .arg_ptr(value)
            .arg_ptr(positions)
            .arg_ptr(table)
            .arg_u32(u32::try_from(rows)?)
            .arg_u32(u32::try_from(prefix_rows)?)
            .arg_u32(u32::try_from(padded)?)
            .launch(self.stream)?;
            self.attention_mqa_padded(query, key, value, out, rows, keys, scores)
        } else {
            self.rope_pair(
                query,
                key.offset(prefix_rows * 256 * 2),
                positions,
                rows,
                8,
                256,
            )?;
            self.attention_mqa_cached(query, key, value, out, rows, keys, capacity_rows, scores)
        }
    }
}
