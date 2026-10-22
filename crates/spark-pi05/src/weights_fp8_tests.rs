// SPDX-License-Identifier: AGPL-3.0-only
use super::*;

fn fixture(expert: bool) -> Manifest {
    let precision = if expert {
        "fp8-vision-expert-bf16"
    } else {
        "fp8-vision-bf16"
    };
    fixture_recipe(precision)
}

fn fixture_recipe(precision: &str) -> Manifest {
    let shapes = expected_shapes(precision).unwrap();
    let mut offset = 0;
    let tensors = shapes
        .iter()
        .map(|(name, &shape)| {
            let entry = Entry {
                name: name.clone(),
                shape,
                data_offset: offset,
                weight_scale: 0.01,
                activation_scale: 0.1,
                packed_sha256: "a".repeat(64),
            };
            offset += (shape[0] * shape[1]) as u64;
            entry
        })
        .collect();
    Manifest {
        format_version: 1,
        model_type: "pi05".into(),
        profile: "pi05_libero".into(),
        precision: precision.into(),
        weight_file: "weights.bin".into(),
        base_checkpoint_sha256: "b".repeat(64),
        nvfp4_manifest_sha256: "c".repeat(64),
        base_tensor_sha256: shapes.keys().map(|n| (n.clone(), "d".repeat(64))).collect(),
        calibration: serde_json::json!({"rows":32}),
        tensors,
    }
}

#[test]
fn inventory_requires_all_and_only_declared_fp8_regions() {
    let expert_only = fixture_recipe("fp8-expert-bf16");
    assert_eq!(expert_only.tensors.len(), 126);
    expert_only.validate(u64::MAX).unwrap();
    for (expert, count) in [(false, 163), (true, 289)] {
        let m = fixture(expert);
        assert_eq!(m.tensors.len(), count);
        m.validate(u64::MAX).unwrap();
    }
    let mut m = fixture(false);
    m.tensors.pop();
    assert!(m.validate(u64::MAX).is_err());
    let mut m = fixture(false);
    m.precision = "fp8-vision-expert-bf16".into();
    assert!(m.validate(u64::MAX).is_err());
    let mut m = fixture(false);
    m.tensors[0].shape = [usize::MAX, usize::MAX];
    assert!(m.validate(u64::MAX).is_err());
    let mut m = fixture(false);
    m.tensors[1].name = m.tensors[0].name.clone();
    assert!(m.validate(u64::MAX).is_err());
    let mut m = fixture(false);
    m.base_tensor_sha256.clear();
    assert!(m.validate(u64::MAX).is_err());
}

#[test]
fn rejects_invalid_scales_spans_paths_and_provenance() {
    for scale in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::from_bits(1)] {
        let mut m = fixture(false);
        m.tensors[0].activation_scale = scale;
        assert!(m.validate(u64::MAX).is_err());
    }
    let mut m = fixture(false);
    m.tensors[0].weight_scale = f32::MAX;
    m.tensors[0].activation_scale = 10.0;
    assert!(m.validate(u64::MAX).is_err());
    let mut m = fixture(false);
    m.tensors[1].data_offset = 0;
    assert!(m.validate(u64::MAX).is_err());
    assert!(fixture(false).validate(1024).is_err());
    let mut m = fixture(false);
    m.weight_file = "../weights.bin".into();
    assert!(m.validate(u64::MAX).is_err());
    let mut m = fixture(false);
    m.nvfp4_manifest_sha256 = "bad".into();
    assert!(m.validate(u64::MAX).is_err());
    let mut m = fixture(false);
    m.calibration = serde_json::Value::Null;
    assert!(m.validate(u64::MAX).is_err());
}

#[test]
fn binding_rejects_different_nvfp4_or_source_weights() {
    let m = fixture(false);
    m.validate_binding(Some(&m.nvfp4_manifest_sha256), &m.base_tensor_sha256)
        .unwrap();
    assert!(
        m.validate_binding(Some(&"e".repeat(64)), &m.base_tensor_sha256)
            .is_err()
    );
    assert!(m.validate_binding(None, &m.base_tensor_sha256).is_err());
    let mut hashes = m.base_tensor_sha256.clone();
    hashes.insert(m.tensors[0].name.clone(), "f".repeat(64));
    assert!(
        m.validate_binding(Some(&m.nvfp4_manifest_sha256), &hashes)
            .is_err()
    );
}

#[test]
fn finite_e4m3_payload_and_hash_are_both_required() {
    let bytes = [0, 0x7e, 0xfe, 0x80];
    let hash = format!("{:x}", Sha256::digest(bytes));
    validate_payload(&bytes, &hash).unwrap();
    assert!(validate_payload(&[0x7f], &format!("{:x}", Sha256::digest([0x7f]))).is_err());
    assert!(validate_payload(&[0xff], &format!("{:x}", Sha256::digest([0xff]))).is_err());
    assert!(validate_payload(&bytes, &"0".repeat(64)).is_err());
}
