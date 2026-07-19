// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! The BLUT dashboard (ADR 0083 M2) — a Leptos CSR app, Rust end-to-end.
//!
//! Talks ONLY to the blut-web sidecar's own `/api` (same origin — the bundle
//! is embedded in and served by that binary), and deserializes the wasm-safe
//! `blut-types` wire types those endpoints emit ([`blut_types::report`]) — the
//! keystone premise: one set of Rust structs on both sides of the wire.
//!
//! Surfaces (deliberately the TUI's read set + the exec bridge):
//! * jobs table (auto-refresh) → per-job `status.jsonl` tail + cancel
//! * a recipe launcher (POST /api/jobs through the rbac exec bridge)
//! * lineage lookup: artifact hash → provenance graph summary + model card
//!
//! Auth: an optional bearer token (the ADR-0095 store) held in memory and
//! attached to every request; the sidecar's custody posture (shared-tenant,
//! restricted-excluded) is enforced SERVER-side — this UI holds no policy.

use leptos::prelude::*;
use leptos::task::spawn_local;

use blut_types::report::{ModelCard, ProvenanceGraph};

/// One row of `GET /api/jobs` — mirrors the engine's `JobSummary` wire shape
/// (tolerant: unknown fields ignored, absent ones default).
#[derive(Clone, Debug, Default, PartialEq, serde::Deserialize)]
struct JobRow {
    #[serde(default)]
    id: String,
    #[serde(default)]
    state: serde_json::Value,
    #[serde(default)]
    pid: Option<u32>,
    #[serde(default)]
    output_name: Option<String>,
    #[serde(default)]
    last_loss: Option<f32>,
    #[serde(default)]
    last_step: Option<u32>,
    #[serde(default)]
    final_loss: Option<f32>,
}

impl JobRow {
    fn state_label(&self) -> String {
        match &self.state {
            serde_json::Value::String(s) => s.clone(),
            other => other.to_string(),
        }
    }
}

async fn api_get(token: &str, path: &str) -> Result<gloo_net::http::Response, String> {
    let mut req = gloo_net::http::Request::get(path);
    if !token.is_empty() {
        req = req.header("Authorization", &format!("Bearer {token}"));
    }
    req.send().await.map_err(|e| e.to_string())
}

async fn api_post(
    token: &str,
    path: &str,
    body: serde_json::Value,
) -> Result<gloo_net::http::Response, String> {
    let mut req = gloo_net::http::Request::post(path).header("Content-Type", "application/json");
    if !token.is_empty() {
        req = req.header("Authorization", &format!("Bearer {token}"));
    }
    req.body(body.to_string())
        .map_err(|e| e.to_string())?
        .send()
        .await
        .map_err(|e| e.to_string())
}

/// Surface an HTTP error body's `error` field (the sidecar's uniform shape).
async fn err_of(res: gloo_net::http::Response) -> String {
    let status = res.status();
    let detail = res
        .json::<serde_json::Value>()
        .await
        .ok()
        .and_then(|v| v.get("error").and_then(|e| e.as_str()).map(String::from))
        .unwrap_or_default();
    format!("HTTP {status} {detail}")
}

fn main() {
    console_error_panic_hook_lite();
    leptos::mount::mount_to_body(App);
}

/// A tiny panic-to-console hook (avoids the console_error_panic_hook dep).
fn console_error_panic_hook_lite() {
    std::panic::set_hook(Box::new(|info| {
        leptos::logging::error!("panic: {info}");
    }));
}

#[component]
fn App() -> impl IntoView {
    let token = RwSignal::new(String::new());
    let flash = RwSignal::new(String::new());

    view! {
        <header class="bar">
            <h1>"BLUT"</h1>
            <span class="sub">"dashboard — blut-web sidecar (ADR 0083)"</span>
            <span class="spacer"></span>
            <input
                type="password"
                placeholder="bearer token (optional on loopback)"
                prop:value=move || token.get()
                on:input=move |ev| token.set(event_target_value(&ev))
            />
        </header>
        <p class="flash">{move || flash.get()}</p>
        <main>
            <Jobs token=token flash=flash />
            <Launcher token=token flash=flash />
            <Lineage token=token />
        </main>
    }
}

#[component]
fn Jobs(token: RwSignal<String>, flash: RwSignal<String>) -> impl IntoView {
    let jobs = RwSignal::new(Vec::<JobRow>::new());
    let selected = RwSignal::new(String::new());
    let events = RwSignal::new(Vec::<String>::new());
    let tick = RwSignal::new(0u64);

    // Auto-refresh loop: re-fetch the jobs table every 5s (and immediately
    // whenever the token changes, since `tick` is only part of the trigger).
    spawn_local(async move {
        loop {
            gloo_timers::future::TimeoutFuture::new(5_000).await;
            tick.update(|t| *t += 1);
        }
    });
    Effect::new(move |_| {
        tick.get();
        let tok = token.get();
        spawn_local(async move {
            if let Ok(res) = api_get(&tok, "/api/jobs").await {
                if res.ok() {
                    if let Ok(list) = res.json::<Vec<JobRow>>().await {
                        jobs.set(list);
                    }
                } else {
                    flash.set(err_of(res).await);
                }
            }
        });
    });

    // Selected-job status tail.
    Effect::new(move |_| {
        let id = selected.get();
        let tok = token.get();
        if id.is_empty() {
            return;
        }
        spawn_local(async move {
            let path = format!("/api/jobs/{id}/status?tail=50");
            if let Ok(res) = api_get(&tok, &path).await
                && res.ok()
                && let Ok(v) = res.json::<serde_json::Value>().await
            {
                let lines = v["events"]
                    .as_array()
                    .map(|a| a.iter().map(|e| e.to_string()).collect())
                    .unwrap_or_default();
                events.set(lines);
            }
        });
    });

    let cancel = move |id: String| {
        let tok = token.get_untracked();
        spawn_local(async move {
            match api_post(
                &tok,
                &format!("/api/jobs/{id}/cancel"),
                serde_json::json!({}),
            )
            .await
            {
                Ok(res) if res.ok() => flash.set(format!("cancel sent to {id}")),
                Ok(res) => flash.set(err_of(res).await),
                Err(e) => flash.set(e),
            }
        });
    };

    view! {
        <section>
            <h2>"Jobs"</h2>
            <table>
                <thead>
                    <tr>
                        <th>"id"</th><th>"state"</th><th>"step"</th><th>"loss"</th><th></th>
                    </tr>
                </thead>
                <tbody>
                    <For each=move || jobs.get() key=|j| j.id.clone() let:j>
                        {
                            let id = j.id.clone();
                            let id2 = j.id.clone();
                            let id3 = j.id.clone();
                            view! {
                                <tr
                                    class:sel=move || selected.get() == id
                                    on:click=move |_| selected.set(id2.clone())
                                >
                                    <td>{j.id.clone()}</td>
                                    <td>{j.state_label()}</td>
                                    <td>{j.last_step.map(|s| s.to_string()).unwrap_or_default()}</td>
                                    <td>{j.final_loss.or(j.last_loss).map(|l| format!("{l:.4}")).unwrap_or_default()}</td>
                                    <td>
                                        <button on:click=move |ev| {
                                            ev.stop_propagation();
                                            cancel(id3.clone());
                                        }>"cancel"</button>
                                    </td>
                                </tr>
                            }
                        }
                    </For>
                </tbody>
            </table>
            <Show when=move || !selected.get().is_empty()>
                <h3>{move || format!("status — {}", selected.get())}</h3>
                <pre class="tail">
                    {move || events.get().join("\n")}
                </pre>
            </Show>
        </section>
    }
}

#[component]
fn Launcher(token: RwSignal<String>, flash: RwSignal<String>) -> impl IntoView {
    let recipe = RwSignal::new(String::new());
    let args = RwSignal::new(String::from("{}"));

    let launch = move |_| {
        let tok = token.get_untracked();
        let name = recipe.get_untracked();
        let parsed: Result<serde_json::Value, _> = serde_json::from_str(&args.get_untracked());
        let body = match parsed {
            Ok(a) => serde_json::json!({ "recipe": name, "args": a }),
            Err(e) => {
                flash.set(format!("args is not valid JSON: {e}"));
                return;
            }
        };
        spawn_local(async move {
            match api_post(&tok, "/api/jobs", body).await {
                Ok(res) if res.ok() => {
                    let v = res.json::<serde_json::Value>().await.unwrap_or_default();
                    flash.set(format!(
                        "launched {} (pid {})",
                        v["recipe"].as_str().unwrap_or("?"),
                        v["pid"]
                    ));
                }
                Ok(res) => flash.set(err_of(res).await),
                Err(e) => flash.set(e),
            }
        });
    };

    view! {
        <section>
            <h2>"Launch"</h2>
            <p class="hint">
                "Runs through the exec bridge: rbac::enforce (operator token) → the CLI, "
                "audited to audit.jsonl."
            </p>
            <input
                placeholder="recipe name"
                prop:value=move || recipe.get()
                on:input=move |ev| recipe.set(event_target_value(&ev))
            />
            <textarea
                rows="3"
                prop:value=move || args.get()
                on:input=move |ev| args.set(event_target_value(&ev))
            ></textarea>
            <button on:click=launch>"run recipe"</button>
        </section>
    }
}

#[component]
fn Lineage(token: RwSignal<String>) -> impl IntoView {
    let hash = RwSignal::new(String::new());
    let summary = RwSignal::new(String::new());
    let card = RwSignal::new(Option::<ModelCard>::None);

    let look = move |_| {
        let tok = token.get_untracked();
        let h = hash.get_untracked();
        spawn_local(async move {
            match api_get(&tok, &format!("/api/lineage/graph/{h}")).await {
                Ok(res) if res.ok() => match res.json::<ProvenanceGraph>().await {
                    Ok(g) => summary.set(format!(
                        "{} nodes, {} edges upstream of {h}",
                        g.nodes.len(),
                        g.edges.len()
                    )),
                    Err(e) => summary.set(format!("bad graph payload: {e}")),
                },
                Ok(res) => summary.set(err_of(res).await),
                Err(e) => summary.set(e),
            }
        });
        let tok = token.get_untracked();
        let h = hash.get_untracked();
        spawn_local(async move {
            card.set(None);
            if let Ok(res) = api_get(&tok, &format!("/api/lineage/card/{h}")).await
                && res.ok()
                && let Ok(c) = res.json::<ModelCard>().await
            {
                card.set(Some(c));
            }
        });
    };

    view! {
        <section>
            <h2>"Lineage"</h2>
            <input
                placeholder="artifact content hash"
                prop:value=move || hash.get()
                on:input=move |ev| hash.set(event_target_value(&ev))
            />
            <button on:click=look>"trace"</button>
            <p>{move || summary.get()}</p>
            <Show when=move || card.get().is_some()>
                <pre class="tail">
                    {move || {
                        card.get()
                            .and_then(|c| serde_json::to_string_pretty(&c).ok())
                            .unwrap_or_default()
                    }}
                </pre>
            </Show>
        </section>
    }
}
