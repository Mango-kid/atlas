// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use axum::{body::Body, http::Request};
use spark_pi05::config::{ACTION_DIM, ACTION_HORIZON, IMAGE_SIZE, STATE_DIM};
use tower::ServiceExt;

fn observation() -> PolicyRequest {
    PolicyRequest {
        image: vec![0; IMAGE_SIZE * IMAGE_SIZE * 3],
        wrist_image: vec![0; IMAGE_SIZE * IMAGE_SIZE * 3],
        state: [0.0; STATE_DIM],
        prompt: "pick up the cup".into(),
        noise: vec![0.0; ACTION_HORIZON * ACTION_DIM],
    }
}

fn post_json(body: impl Into<Body>) -> Request<Body> {
    Request::post("/v1/actions")
        .header("content-type", "application/json")
        .body(body.into())
        .expect("request")
}

#[tokio::test]
async fn health_reflects_worker_liveness() {
    let (sender, receiver) = mpsc::channel(1);
    let app = router(sender, 1024, "atlas-native-bf16");
    let request = || {
        Request::get("/healthz")
            .body(Body::empty())
            .expect("request")
    };
    assert_eq!(
        app.clone()
            .oneshot(request())
            .await
            .expect("response")
            .status(),
        StatusCode::OK
    );
    drop(receiver);
    assert_eq!(
        app.oneshot(request()).await.expect("response").status(),
        StatusCode::SERVICE_UNAVAILABLE
    );
}

#[tokio::test]
async fn rejects_malformed_or_oversized_json_before_enqueueing() {
    let (sender, mut receiver) = mpsc::channel(1);
    let app = router(sender, 1024, "atlas-native-bf16");
    assert_eq!(
        app.clone()
            .oneshot(post_json("{}"))
            .await
            .expect("response")
            .status(),
        StatusCode::BAD_REQUEST
    );
    assert_eq!(
        app.oneshot(post_json("x".repeat(1025)))
            .await
            .expect("response")
            .status(),
        StatusCode::PAYLOAD_TOO_LARGE
    );
    assert!(receiver.try_recv().is_err());
}

#[tokio::test]
async fn rejects_invalid_image_shape_before_enqueueing() {
    let (sender, mut receiver) = mpsc::channel(1);
    let app = router(sender, 2_097_152, "atlas-native-bf16");
    let mut request = observation();
    request.image.pop();
    let response = app
        .oneshot(post_json(serde_json::to_vec(&request).expect("JSON")))
        .await
        .expect("response");
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert!(receiver.try_recv().is_err());
}

#[tokio::test]
async fn saturated_queue_returns_429_without_waiting() {
    let (sender, _receiver) = mpsc::channel(1);
    let (reply, _result) = oneshot::channel();
    sender
        .try_send(Job {
            request: observation(),
            reply,
        })
        .expect("queue first job");
    let app = router(sender, 2_097_152, "atlas-native-bf16");
    let response = app
        .oneshot(post_json(serde_json::to_vec(&observation()).expect("JSON")))
        .await
        .expect("response");
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
}

#[tokio::test]
async fn worker_failure_is_500_and_does_not_disclose_internal_error() {
    let (sender, mut receiver) = mpsc::channel::<Job>(1);
    let worker = tokio::spawn(async move {
        let job = receiver.recv().await.expect("job");
        let _ = job
            .reply
            .send(Err(anyhow::anyhow!("private filesystem path")));
    });
    let app = router(sender, 2_097_152, "atlas-native-bf16");
    let response = app
        .oneshot(post_json(serde_json::to_vec(&observation()).expect("JSON")))
        .await
        .expect("response");
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    let body = axum::body::to_bytes(response.into_body(), 1024)
        .await
        .expect("body");
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&body).expect("JSON"),
        serde_json::json!({"error":"policy inference failed"})
    );
    worker.await.expect("worker");
}

#[tokio::test]
async fn action_response_preserves_shape_and_timing() {
    let (sender, mut receiver) = mpsc::channel::<Job>(1);
    let worker = tokio::spawn(async move {
        let job = receiver.recv().await.expect("job");
        let _ = job.reply.send(Ok(spark_pi05::PolicyResponse {
            actions: vec![[0.25; 7]; ACTION_HORIZON],
            timing_ms: 42.5,
            backend: "atlas-native-fp8-nvfp4",
        }));
    });
    let app = router(sender, 2_097_152, "atlas-native-fp8-nvfp4");
    let response = app
        .oneshot(post_json(serde_json::to_vec(&observation()).expect("JSON")))
        .await
        .expect("response");
    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), 4096)
        .await
        .expect("body");
    let result: serde_json::Value = serde_json::from_slice(&body).expect("JSON");
    assert_eq!(
        result["actions"].as_array().expect("actions").len(),
        ACTION_HORIZON
    );
    assert_eq!(
        result["actions"][0],
        serde_json::to_value([0.25; 7]).expect("action row")
    );
    assert_eq!(result["timing_ms"], 42.5);
    assert_eq!(result["backend"], "atlas-native-fp8-nvfp4");
    worker.await.expect("worker");
}

#[tokio::test]
async fn health_reports_precision_of_loaded_policy() {
    for backend in [
        "atlas-native-bf16",
        "atlas-native-nvfp4",
        "atlas-native-fp8-nvfp4",
    ] {
        let (sender, _receiver) = mpsc::channel(1);
        let app = router(sender, 1024, backend);
        let response = app
            .oneshot(
                Request::get("/healthz")
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), 1024)
            .await
            .expect("body");
        let result: serde_json::Value = serde_json::from_slice(&body).expect("JSON");
        assert_eq!(result["backend"], backend);
        assert_eq!(result["profile"], "pi05_libero");
    }
}
