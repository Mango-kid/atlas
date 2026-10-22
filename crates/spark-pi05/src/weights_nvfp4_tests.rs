// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

fn fixture() -> Manifest {
    let mut offset = 0u64;
    let tensors = expected_shapes()
        .unwrap()
        .into_iter()
        .map(|(name, shape)| {
            let mut entry = Entry {
                name,
                shape,
                data_offset: offset,
                scale_offset: 0,
                weight_global_scale: 0.01,
                activation_global_scale: 0.1,
                packed_sha256: "0".repeat(64),
                scales_sha256: "0".repeat(64),
            };
            let (packed, scales) = entry.lengths();
            entry.scale_offset = offset + packed as u64;
            offset += (packed + scales) as u64;
            entry
        })
        .collect();
    Manifest {
        format_version: 1,
        model_type: "pi05".into(),
        profile: "pi05_libero".into(),
        precision: "nvfp4-prefix-bf16".into(),
        weight_file: "weights.bin".into(),
        base_checkpoint_sha256: "0".repeat(64),
        base_tensor_sha256: BTreeMap::new(),
        calibration: serde_json::json!({"rows":32}),
        tensors,
    }
}

#[test]
fn rejects_wrong_inventory_shapes_scales_and_spans() {
    fixture().validate(u64::MAX).unwrap();
    let mut m = fixture();
    m.tensors.pop();
    assert!(m.validate(u64::MAX).is_err());
    let mut m = fixture();
    m.tensors[1].name = m.tensors[0].name.clone();
    assert!(m.validate(u64::MAX).is_err());
    let mut m = fixture();
    m.tensors[0].shape[0] += 1;
    assert!(m.validate(u64::MAX).is_err());
    let mut m = fixture();
    m.tensors[0].activation_global_scale = f32::NAN;
    assert!(m.validate(u64::MAX).is_err());
    let mut m = fixture();
    m.tensors[0].scale_offset = 0;
    assert!(m.validate(u64::MAX).is_err());
    assert!(fixture().validate(10).is_err());
    let mut m = fixture();
    m.weight_file = "../weights.bin".into();
    assert!(m.validate(u64::MAX).is_err());
}

#[test]
fn validates_consumed_bytes_and_rejects_unknown_format() {
    let hash = format!("{:x}", Sha256::digest(b"reference"));
    verify_digest(b"reference", &hash, "test").unwrap();
    assert!(verify_digest(b"changed", &hash, "test").is_err());
    let mut m = fixture();
    m.precision = "nvfp4-all-linears".into();
    assert!(m.validate(u64::MAX).is_err());
    let mut m = fixture();
    m.format_version = 2;
    assert!(m.validate(u64::MAX).is_err());
}

struct MemorySource {
    tensors: BTreeMap<String, Vec<u8>>,
}

impl WeightSource for MemorySource {
    fn names(&self) -> Vec<String> {
        self.tensors.keys().cloned().collect()
    }

    fn read(&mut self, name: &str) -> Result<TensorData> {
        let bytes = self
            .tensors
            .get(name)
            .context("missing test tensor")?
            .clone();
        Ok(TensorData {
            name: name.into(),
            shape: vec![bytes.len() / 2],
            bytes,
        })
    }
}

#[test]
fn retained_hash_inventory_must_match_consumed_source_tensors_exactly() {
    let mut manifest = fixture();
    let retained_name = "action_out_proj.bias".to_string();
    manifest
        .base_tensor_sha256
        .insert(retained_name.clone(), "0".repeat(64));
    let mut source = MemorySource {
        tensors: manifest
            .tensors
            .iter()
            .map(|entry| (entry.name.clone(), vec![]))
            .collect(),
    };
    source.tensors.insert(retained_name.clone(), vec![0; 4]);
    source.tensors.insert(UNUSED.into(), vec![]);
    manifest
        .validate_source(&source)
        .expect("complete source inventory");

    manifest.base_tensor_sha256.remove(&retained_name);
    assert!(
        manifest.validate_source(&source).is_err(),
        "missing retained hash"
    );
    manifest
        .base_tensor_sha256
        .insert(retained_name, "0".repeat(64));
    manifest
        .base_tensor_sha256
        .insert("unavailable.weight".into(), "0".repeat(64));
    assert!(
        manifest.validate_source(&source).is_err(),
        "hash for absent tensor"
    );
    manifest.base_tensor_sha256.remove("unavailable.weight");
    source.tensors.insert("unhashed.weight".into(), vec![0; 4]);
    assert!(
        manifest.validate_source(&source).is_err(),
        "unhashed source tensor"
    );
    source.tensors.remove("unhashed.weight");
    source.tensors.remove(&manifest.tensors[0].name);
    assert!(
        manifest.validate_source(&source).is_err(),
        "missing prefix source tensor"
    );
}

#[test]
fn verified_base_returns_matching_bytes_and_rejects_corruption_or_unhashed_reads() {
    let bytes = vec![0x80, 0x3f, 0x00, 0x40];
    let digest = format!("{:x}", Sha256::digest(&bytes));
    let hashes = BTreeMap::from([("kept.weight".into(), digest)]);
    let mut source = MemorySource {
        tensors: BTreeMap::from([
            ("kept.weight".into(), bytes.clone()),
            ("unhashed.weight".into(), bytes.clone()),
        ]),
    };
    {
        let mut verified = VerifiedBase {
            source: &mut source,
            hashes: &hashes,
        };
        assert_eq!(verified.names(), vec!["kept.weight"]);
        let tensor = verified.read("kept.weight").expect("matching digest");
        assert_eq!(tensor.bytes, bytes);
        assert_eq!(tensor.shape, vec![2]);
        assert!(verified.read("unhashed.weight").is_err());
    }
    source.tensors.get_mut("kept.weight").unwrap()[0] ^= 1;
    let mut verified = VerifiedBase {
        source: &mut source,
        hashes: &hashes,
    };
    let error = verified
        .read("kept.weight")
        .err()
        .expect("corruption rejected");
    assert!(error.to_string().contains("SHA-256 mismatch"));
}

#[test]
fn finite_global_product_does_not_allow_quantizer_denominator_overflow() {
    let mut manifest = fixture();
    let entry = &mut manifest.tensors[0];
    entry.activation_global_scale = f32::MAX / 2.0;
    entry.weight_global_scale = f32::MIN_POSITIVE;
    assert!((entry.activation_global_scale * entry.weight_global_scale).is_finite());
    assert!(!(6.0 * entry.activation_global_scale).is_finite());
    let error = manifest
        .validate(u64::MAX)
        .expect_err("unsafe quantizer scale");
    assert!(
        error
            .to_string()
            .contains("activation scale exceeds FP32 quantizer range")
    );
}
