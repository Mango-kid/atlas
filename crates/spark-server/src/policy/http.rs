// SPDX-License-Identifier: AGPL-3.0-only

use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, State, rejection::JsonRejection},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use spark_pi05::preprocess::PolicyRequest;
use tokio::sync::{mpsc, oneshot};

use super::runtime::Job;

#[derive(Clone)]
struct PolicyState {
    sender: mpsc::Sender<Job>,
    backend: &'static str,
}

pub(super) fn router(
    sender: mpsc::Sender<Job>,
    max_body_bytes: usize,
    backend: &'static str,
) -> Router {
    Router::new()
        .route("/healthz", get(health))
        .route("/v1/actions", post(actions))
        .layer(DefaultBodyLimit::max(max_body_bytes))
        .with_state(PolicyState { sender, backend })
}

async fn health(State(state): State<PolicyState>) -> Response {
    if state.sender.is_closed() {
        return error(StatusCode::SERVICE_UNAVAILABLE, "policy worker unavailable");
    }
    Json(serde_json::json!({"status":"ok", "backend":state.backend, "profile":"pi05_libero"}))
        .into_response()
}

async fn actions(
    State(state): State<PolicyState>,
    request: Result<Json<PolicyRequest>, JsonRejection>,
) -> Response {
    let request = match request {
        Ok(Json(request)) => request,
        Err(rejection) => {
            let status = if rejection.status() == StatusCode::PAYLOAD_TOO_LARGE {
                StatusCode::PAYLOAD_TOO_LARGE
            } else {
                StatusCode::BAD_REQUEST
            };
            return error(status, &rejection.body_text());
        }
    };
    if let Err(reason) = request.validate() {
        return error(StatusCode::BAD_REQUEST, &reason.to_string());
    }
    let (reply, result) = oneshot::channel();
    match state.sender.try_send(Job { request, reply }) {
        Ok(()) => {}
        Err(mpsc::error::TrySendError::Full(_)) => {
            return error(StatusCode::TOO_MANY_REQUESTS, "policy queue is full");
        }
        Err(mpsc::error::TrySendError::Closed(_)) => {
            return error(StatusCode::SERVICE_UNAVAILABLE, "policy worker unavailable");
        }
    }
    match result.await {
        Ok(Ok(response)) => Json(response).into_response(),
        Ok(Err(reason)) => {
            tracing::error!(error = %reason, "policy inference failed");
            error(StatusCode::INTERNAL_SERVER_ERROR, "policy inference failed")
        }
        Err(_) => error(StatusCode::SERVICE_UNAVAILABLE, "policy worker unavailable"),
    }
}

fn error(status: StatusCode, message: &str) -> Response {
    (status, Json(serde_json::json!({"error":message}))).into_response()
}

#[cfg(test)]
mod tests;
