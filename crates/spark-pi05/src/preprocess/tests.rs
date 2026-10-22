// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

fn stats() -> ActionNormalizer {
    ActionNormalizer::from_json(
        br#"{"norm_stats":{"actions":{"q01":[-2,-2,-2,-2,-2,-2,-1],"q99":[2,2,2,2,2,2,1]}}}"#,
    )
    .expect("valid statistics")
}

#[test]
fn patch_layout_matches_channel_first_conv2d() {
    let mut rgb = vec![0; IMAGE_SIZE * IMAGE_SIZE * 3];
    rgb[0] = 255;
    rgb[1] = 128;
    rgb[3] = 64;
    rgb[PATCH_SIZE * 3] = 32;
    rgb[(IMAGE_SIZE - 1) * IMAGE_SIZE * 3 + (IMAGE_SIZE - 1) * 3 + 2] = 255;
    let patches = image_patches(&rgb).expect("image");
    assert_eq!(patches.len(), IMAGE_SIZE * IMAGE_SIZE * 3);
    assert_eq!(patches[0], 1.0);
    assert_eq!(patches[1], 64.0 / 255.0 * 2.0 - 1.0);
    assert_eq!(patches[PATCH_SIZE * PATCH_SIZE], 128.0 / 255.0 * 2.0 - 1.0);
    assert_eq!(
        patches[PATCH_SIZE * PATCH_SIZE * 3],
        32.0 / 255.0 * 2.0 - 1.0
    );
    assert_eq!(*patches.last().expect("last pixel"), 1.0);
    assert!(image_patches(&rgb[..rgb.len() - 1]).is_err());
}

#[test]
fn action_inverse_uses_quantiles_and_preserves_out_of_range_values() {
    let mut normalized = vec![0.0; ACTION_HORIZON * ACTION_DIM];
    normalized[0] = -1.0;
    normalized[1] = 1.0;
    normalized[2] = 2.0;
    normalized[6] = -1.0;
    // Padded dimensions must not affect the seven actuator outputs.
    normalized[7] = 123.0;
    let actions = stats().decode(&normalized).expect("actions");
    assert_eq!(actions.len(), ACTION_HORIZON);
    assert_eq!(actions[0][0], -2.0);
    assert!((actions[0][1] - 2.000_001).abs() < 1e-12);
    assert!((actions[0][2] - 4.000_001_5).abs() < 1e-12);
    assert_eq!(actions[0][6], -1.0);
    assert!(
        stats()
            .decode(&normalized[..ACTION_HORIZON * ACTION_DIM - 1])
            .is_err()
    );
    normalized[0] = f32::NAN;
    assert!(stats().decode(&normalized).is_err());
}

#[test]
fn reject_incompatible_normalization_assets() {
    for json in [
        r#"{"norm_stats":{"actions":{"q01":[0],"q99":[1]}}}"#,
        r#"{"norm_stats":{"actions":{"q01":[1,0,0,0,0,0,0],"q99":[0,1,1,1,1,1,1]}}}"#,
        r#"{"norm_stats":{"actions":{"mean":[0],"std":[1]}}}"#,
    ] {
        assert!(ActionNormalizer::from_json(json.as_bytes()).is_err());
    }
}

#[test]
fn prompt_cleaning_preserves_case_and_internal_spaces() {
    assert_eq!(clean_prompt(" \tPick_up\nthe  Cup\r\n"), "Pick up the  Cup");
    assert_eq!(clean_prompt("\u{2003}é你好\u{2003}"), "é你好");
}

#[test]
fn request_requires_explicit_fields_and_valid_finite_tensors() {
    assert!(serde_json::from_str::<PolicyRequest>("{}").is_err());
    let mut request = PolicyRequest {
        image: vec![0; IMAGE_SIZE * IMAGE_SIZE * 3],
        wrist_image: vec![255; IMAGE_SIZE * IMAGE_SIZE * 3],
        state: [0.0; STATE_DIM],
        prompt: "Pick up the cup".into(),
        noise: vec![0.0; ACTION_HORIZON * ACTION_DIM],
    };
    assert!(request.validate().is_ok());
    request.state[0] = f64::INFINITY;
    assert!(request.validate().is_err());
    request.state[0] = 0.0;
    request.noise[0] = f32::NAN;
    assert!(request.validate().is_err());
}

/// Goldens from upstream SentencePiece's Python binding and the OpenPi prompt
/// transform, using the original big_vision/paligemma_tokenizer.model asset.
#[test]
#[ignore = "requires PI05_TOKENIZER_MODEL pointing to original SentencePiece asset"]
fn tokenizer_matches_openpi_sentencepiece_goldens() {
    let path = std::env::var("PI05_TOKENIZER_MODEL").expect("PI05_TOKENIZER_MODEL");
    let bytes = std::fs::read(path).expect("tokenizer model");
    let preprocessor = Preprocessor::from_bytes(
        &bytes,
        br#"{"norm_stats":{"actions":{"q01":[-2,-2,-2,-2,-2,-2,-1],"q99":[2,2,2,2,2,2,1]}}}"#,
    )
    .expect("assets");
    for (prompt, expected) in [
        ("do something", vec![2, 1042, 2775, 108]),
        ("  Pick_up\nthe mug  ", vec![2, 21585, 908, 573, 24464, 108]),
        ("", vec![2, 108]),
        (
            "Café 紅色のカップを拾う 🤖",
            vec![
                2, 86769, 59715, 67688, 63569, 235432, 239006, 235480, 235248, 243349, 108,
            ],
        ),
        ("a  b\tc", vec![2, 235250, 139, 235268, 226, 235260, 108]),
    ] {
        assert_eq!(preprocessor.tokenize(prompt).expect("tokenize"), expected);
    }
    let long = "hello ".repeat(MAX_PROMPT_TOKENS * 2);
    let tokens = preprocessor.tokenize(&long).expect("long prompt");
    assert_eq!(tokens.len(), MAX_PROMPT_TOKENS);
    assert_eq!(tokens[0], 2);
    assert!(
        preprocessor
            .tokenize(&"x".repeat(MAX_PROMPT_BYTES + 1))
            .is_err()
    );
}
