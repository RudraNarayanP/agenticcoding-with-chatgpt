//! OpenRouter transport: a cheap model as the *hands*, never the brain.
//!
//! The division of labour this exists to enforce. ChatGPT (especially Pro, which
//! is browser-only and can't use MCP) is the planner and reviewer: it decides
//! what should change and judges whether it worked. The edit-loop model -- the
//! thing that emits `write_file` calls one keystroke at a time -- is a free-tier
//! OpenRouter model, because that work needs no reasoning budget and burning a
//! paid plan's quota on it is pure waste.
//!
//! Two properties fall out of that and shape everything here:
//!
//! - **Fresh context per chunk.** OpenRouter is a stateless HTTP API, so unlike
//!   the browser channel -- where the conversation accumulates server-side and a
//!   run only sends the newest tool results -- every request must resend its own
//!   history. Rather than let that grow without bound, a chunk gets its own
//!   message list, seeded with just enough of the plan to do its steps. When the
//!   chunk ends, the history is thrown away and the next one starts clean. That
//!   is the point, not a limitation: it is what keeps a fifty-tool-call task from
//!   dying of context rot.
//!
//! - **No new dependency.** This crate deliberately avoids an HTTP client:
//!   `jsonschema` has its default features off precisely because they would pull
//!   `reqwest` and a TLS stack, and `tiny_http` is a server. So requests go out
//!   through `curl`, which ships with Windows 10+ and macOS, using the bounded
//!   runner in [`crate::util`]. The JSON body is passed as a file rather than an
//!   argument because a chunk's prompt routinely exceeds Windows' ~32 KB command
//!   line limit.

use crate::protocol::{self, Reply, ToolCall, ToolResult, ToolSpec};
use anyhow::{bail, Context, Result};
use serde::Deserialize;
use serde_json::json;
use std::path::{Path, PathBuf};

/// A chat message in OpenRouter's `/chat/completions` shape.
#[derive(Debug, Clone, serde::Serialize, Deserialize, PartialEq)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn system(text: impl Into<String>) -> Self {
        Message {
            role: "system".into(),
            content: text.into(),
        }
    }
    pub fn user(text: impl Into<String>) -> Self {
        Message {
            role: "user".into(),
            content: text.into(),
        }
    }
    pub fn assistant(text: impl Into<String>) -> Self {
        Message {
            role: "assistant".into(),
            content: text.into(),
        }
    }
}

/// Free-tier models worth using as an executor, best-first.
///
/// Order is by *observed behaviour*, not by reputation, because being listed in
/// `/api/v1/models` turned out not to mean being servable: at the time this was
/// written `qwen/qwen3.8-27b:free`, `z-ai/glm-5.2:free` and
/// `google/gemma-4-31b-it:free` all answered "Provider returned error", and
/// `thinkingmachines/*` is gated on age verification. `poolside/laguna-s-2.1:free`
/// was the first that both served and obeyed a "reply with only X" instruction --
/// the exact discipline this protocol needs. Nemotron serves but narrates.
///
/// Even this order decays, so [`pick_usable`] probes before choosing rather than
/// trusting the list.
pub const PREFERRED_FREE_MODELS: &[&str] = &[
    "poolside/laguna-s-2.1:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "qwen/qwen3.8-27b:free",
    "z-ai/glm-5.2:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
];

/// Back-compat name for tests and error text: the head of the shortlist.
pub const DEFAULT_MODEL: &str = PREFERRED_FREE_MODELS[0];

/// First candidate for which `probe` succeeds.
///
/// Injectable as a closure precisely so the *selection* is unit-testable without
/// a key or a network: the real probe makes a completion, the test probe is a
/// HashSet. Free-tier providers flake independently of OpenRouter, so "is it
/// listed" is the wrong question; "does it answer" is the right one.
pub fn pick_usable<'a, F>(candidates: &[&'a str], mut probe: F) -> Option<&'a str>
where
    F: FnMut(&str) -> bool,
{
    candidates.iter().copied().find(|m| probe(m))
}

/// The live catalogue of free-tier model ids, or `None` if it could not be
/// reached. Keyless and unrelated to chatgpt.com, so this costs no plan quota.
pub fn free_models() -> Option<Vec<String>> {
    let mut cmd = match crate::util::which("curl") {
        Some(p) => crate::util::command_at(&p),
        None => return None,
    };
    cmd.args(["--silent", "--show-error", "--max-time", "20"]);
    cmd.arg("https://openrouter.ai/api/v1/models");
    let out = crate::util::run_bounded(&mut cmd, 30.0).ok()?;
    if !out.status.success() {
        return None;
    }
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).ok()?;
    let ids: Vec<String> = v
        .get("data")?
        .as_array()?
        .iter()
        .filter_map(|m| m.get("id").and_then(|i| i.as_str()))
        .filter(|i| i.ends_with(":free"))
        .map(str::to_string)
        .collect();
    if ids.is_empty() {
        None
    } else {
        Some(ids)
    }
}

/// One small completion, to ask a model "are you serving right now".
///
/// The budget is 64 tokens, not a tiny handful, and that number is load-bearing:
/// several free models reason before they emit, so a tight cap starves them into
/// `Provider returned error` and a 4-token probe reports a perfectly usable model
/// as dead. Measured on `poolside/laguna-s-2.1:free`: fails at 4 and 16, answers
/// at 64. The retry exists for the same reason -- free providers flap.
///
/// Probing rather than trusting the model list is deliberate: the list reflects
/// what OpenRouter offers, not what is answering at this moment.
fn serves(api_key: &str, model: &str) -> bool {
    let Some(curl) = crate::util::which("curl") else { return false };
    let mut cmd = crate::util::command_at(&curl);
    cmd.args(["--silent", "--show-error", "--max-time", "25", "--request", "POST"])
        .args(["--header", &format!("Authorization: Bearer {api_key}"), "--header", "Content-Type: application/json"])
        .args([
            "--data-binary",
            &format!(
                r#"{{"model":"{model}","messages":[{{"role":"user","content":"Reply with only: ok"}}],"max_tokens":64,"temperature":0}}"#
            ),
        ])
        .arg("https://openrouter.ai/api/v1/chat/completions");
    let answer = |cmd: &mut std::process::Command| match crate::util::run_bounded(cmd, 60.0) {
        Ok(out) if out.status.success() => {
            let v = serde_json::from_slice::<serde_json::Value>(&out.stdout).ok()?;
            // An `error` object with no choices is a provider failure, and a model
            // that returns empty content is no use as an executor either.
            (v.pointer("/error").is_none()
                && v.pointer("/choices/0/message/content")
                    .and_then(|c| c.as_str())
                    .is_some_and(|c| !c.trim().is_empty()))
            .then_some(())
        }
        _ => None,
    };
    answer(&mut cmd).is_some() || answer(&mut cmd).is_some()
}

/// Pick the executor model by asking, not by trusting a name.
///
/// The shortlist is probed in order -- at most a handful of 4-token calls, and
/// only when the caller did not name a model -- then anything else the catalogue
/// lists as free is taken unprobed rather than looping the whole catalogue. If
/// nothing answers, the head of the shortlist is returned with a warning: better
/// a loud guess than a silent hang.
fn resolve_model(api_key: &str) -> String {
    let catalogue = free_models();
    let mut candidates: Vec<String> = match &catalogue {
        Some(cat) => PREFERRED_FREE_MODELS
            .iter()
            .filter(|m| cat.iter().any(|c| c.as_str() == **m))
            .map(|m| (*m).to_string())
            .collect(),
        None => PREFERRED_FREE_MODELS.iter().map(|m| (*m).to_string()).collect(),
    };
    if let Some(cat) = &catalogue {
        // Gathered first: the filter would otherwise hold an immutable borrow on
        // `candidates` across the push that mutates it.
        let extra: Vec<String> = cat
            .iter()
            .filter(|c| !candidates.iter().any(|k| k == *c))
            .cloned()
            .collect();
        candidates.extend(extra);
    }
    let probe_limit = PREFERRED_FREE_MODELS.len();
    let mut tried = 0usize;
    let refs: Vec<&str> = candidates.iter().map(String::as_str).collect();
    let winner = pick_usable(&refs, |m| {
        tried += 1;
        if tried > probe_limit {
            // Past the shortlist, stop probing and just take it.
            return false;
        }
        let ok = serves(api_key, m);
        if !ok {
            eprintln!("  {m} did not answer; trying the next free model");
        }
        ok
    });
    match winner {
        Some(m) => m.to_string(),
        None => {
            eprintln!(
                "warning: no preferred free model answered; using {DEFAULT_MODEL} unverified.                  Name one with --exec-model if it is rejected."
            );
            DEFAULT_MODEL.to_string()
        }
    }
}

/// Everything needed to talk to OpenRouter.
pub struct Client {
    pub api_key: String,
    pub model: String,
    /// Wall-clock budget for one completion. Free tiers queue and go slowly;
    /// a large `write_file` reply also takes real time to generate.
    pub timeout_secs: u64,
}

/// Resolve the API key, or explain how to get one.
///
/// Checked in env then file, never generated: this is a credential for a third
/// party, and there is no sensible local stand-in.
pub fn api_key() -> Result<String> {
    if let Ok(k) = std::env::var("OPENROUTER_API_KEY") {
        let k = k.trim().to_string();
        if !k.is_empty() {
            return Ok(k);
        }
    }
    let path =
        key_path().context("no home directory to look for ~/.chatgpt-use/openrouter.key in")?;
    if let Ok(raw) = std::fs::read_to_string(&path) {
        let k = raw.trim().to_string();
        if !k.is_empty() {
            return Ok(k);
        }
    }
    bail!(
        "no OpenRouter API key. Free-tier models still require one -- get it from \
         https://openrouter.ai/keys, then either set OPENROUTER_API_KEY or write it to \
         {}\n  (this tool only ever sends the plan and file contents you approved to it)",
        path.display()
    )
}

/// Refuse anything that is not a free-tier model.
///
/// The premise of this command is that the executor costs nothing, and the
/// account it was first run against is *not* flagged `is_free_tier`, so nothing
/// on OpenRouter's side would stop a paid model: a mistyped `--exec-model` or a
/// dropped `:free` suffix would bill real money mid-task, silently, in a loop.
/// Enforcing the suffix here is the only thing between that typo and a charge.
fn require_free(model: &str) -> Result<()> {
    if model.ends_with(":free") {
        return Ok(());
    }
    bail!(
        "refusing to run `{model}`: it is not a free-tier model. This tool only executes on          models whose id ends in `:free`, and `{model}` does not. Drop --exec-model to have one          resolved from OpenRouter's live free catalogue, or name a :free model explicitly."
    )
}

impl Client {
    pub fn new(model: Option<String>, timeout_secs: u64) -> Result<Self> {
        let explicit = model
            .filter(|m| !m.trim().is_empty())
            .or_else(|| std::env::var("OPENROUTER_MODEL").ok())
            .filter(|m| !m.trim().is_empty());
        // A paid model is refused before any network call at all.
        if let Some(m) = &explicit {
            require_free(m)?;
        }
        let api_key = api_key()?;
        let model = match explicit {
            Some(m) => m,
            None => resolve_model(&api_key),
        };
        require_free(&model)?;
        Ok(Client { api_key, model, timeout_secs })
    }

    /// One completion. Appends the reply to `history` so the caller's context
    /// grows the way an agent loop needs it to.
    pub fn complete(&self, history: &mut Vec<Message>) -> Result<String> {
        let body = self.request_body(history);
        let text = self.post(&body)?;
        let reply = parse_completion(&text)
            .with_context(|| format!("parsing OpenRouter response (model {})", self.model))?;
        history.push(Message::assistant(reply.clone()));
        Ok(reply)
    }

    /// The request JSON. Kept as a pure function so the shape is unit-tested
    /// without a key or a network.
    fn request_body(&self, history: &[Message]) -> serde_json::Value {
        json!({
            "model": self.model,
            "messages": history,
            // The protocol is text, so temperature is load-bearing: 0.2 keeps a
            // cheap model literal about file contents instead of "improving" them.
            "temperature": 0.2,
            "max_tokens": 4096,
        })
    }

    /// POST `body` with curl, passing it through a temp file.
    fn post(&self, body: &serde_json::Value) -> Result<String> {
        let dir = std::env::temp_dir().join(format!("cgu-or-{}", std::process::id()));
        std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
        let file = dir.join("body.json");
        std::fs::write(&file, body.to_string())
            .with_context(|| format!("writing {}", file.display()))?;

        let mut cmd = match crate::util::which("curl") {
            Some(p) => crate::util::command_at(&p),
            None => bail!(
                "curl is required to reach OpenRouter and was not found on PATH \
                 (it ships with Windows 10+ and macOS)"
            ),
        };
        cmd.args([
            "--silent",
            "--show-error",
            "--no-location",
            // Never let a slow or captive network hang the run: this is the
            // failure mode that cost a live request before the transport was
            // bounded.
            "--max-time",
        ])
        .arg(self.timeout_secs.to_string())
        .args([
            "--request",
            "POST",
            "--header",
            &format!("Authorization: Bearer {}", self.api_key),
            "--header",
            "Content-Type: application/json",
        ])
        .arg("--data-binary")
        .arg(format!("@{}", file.display()))
        .arg("https://openrouter.ai/api/v1/chat/completions");

        let out = crate::util::run_bounded(&mut cmd, self.timeout_secs as f64 + 15.0)
            .context("running curl for OpenRouter")?;
        let _ = std::fs::remove_dir_all(&dir);

        let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
        if !out.status.success() {
            let stderr = String::from_utf8_lossy(&out.stderr);
            bail!(
                "OpenRouter request failed (curl exit {:?}): {}",
                out.status.code(),
                stderr.trim().chars().take(300).collect::<String>()
            );
        }
        // A 4xx still arrives as a JSON object; surfacing its message is how a
        // bad key or an exhausted free tier reads honestly instead of as a parse
        // failure.
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&stdout) {
            if let Some(msg) = v.pointer("/error/message").and_then(|m| m.as_str()) {
                bail!(
                    "OpenRouter refused the request: {} (model {})",
                    msg.chars().take(300).collect::<String>(),
                    self.model
                );
            }
        }
        Ok(stdout)
    }
}

/// Pull the assistant text out of a `/chat/completions` response.
///
/// Tolerates the two shapes seen in the wild: `choices[0].message.content` and
/// the older `choices[0].text`.
fn parse_completion(raw: &str) -> Result<String> {
    let v: serde_json::Value = serde_json::from_str(raw.trim()).with_context(|| {
        format!(
            "not JSON: {}",
            raw.trim().chars().take(200).collect::<String>()
        )
    })?;
    let choice = v.get("choices").and_then(|c| c.get(0)).with_context(|| {
        format!(
            "no choices in response: {}",
            raw.trim().chars().take(200).collect::<String>()
        )
    })?;
    let content = choice
        .pointer("/message/content")
        .and_then(|c| c.as_str())
        .or_else(|| choice.get("text").and_then(|t| t.as_str()))
        .context("choice carried no content")?;
    Ok(content.to_string())
}

/// One chunk of a plan: the steps this executor session is responsible for.
#[derive(Debug, Clone)]
pub struct Chunk {
    pub index: usize,
    pub steps: Vec<crate::delegation::PlanStep>,
}

/// Split plan steps into chunks of at most `size`, preserving order.
///
/// Bounded by step count rather than tokens because the step is the unit the
/// planner actually committed to, and `success_criteria` is per step -- so a
/// chunk boundary is also the only place a result can be honestly checked.
pub fn chunk_steps(steps: &[crate::delegation::PlanStep], size: usize) -> Vec<Chunk> {
    if size == 0 {
        // A zero chunk would loop forever.
        return Vec::new();
    }
    steps
        .chunks(size)
        .enumerate()
        .map(|(i, slice)| Chunk {
            index: i,
            steps: slice.to_vec(),
        })
        .collect()
}

/// Seed messages for one chunk: the tool protocol, the plan's fixed context, and
/// only this chunk's steps, plus what earlier chunks achieved.
pub fn chunk_messages(
    specs: &[ToolSpec],
    goal: &str,
    acceptance: &[String],
    do_not_do: &[String],
    chunk: &Chunk,
    completed: &[String],
) -> Vec<Message> {
    let mut brief = String::new();
    brief.push_str(&format!(
        "You are an EXECUTOR. A planner has already decided what to do; your only job is to \
         carry out the steps below by calling tools. Do not redesign, do not add features, do \
         not explain -- act.\n\nGOAL: {goal}\n"
    ));
    if !acceptance.is_empty() {
        brief.push_str(&format!(
            "\nACCEPTANCE CRITERIA:\n- {}\n",
            acceptance.join("\n- ")
        ));
    }
    if !do_not_do.is_empty() {
        brief.push_str(&format!("\nDO NOT DO:\n- {}\n", do_not_do.join("\n- ")));
    }
    if !completed.is_empty() {
        brief.push_str(&format!(
            "\nALREADY COMPLETED BY EARLIER STEPS (do not redo these):\n- {}\n",
            completed.join("\n- ")
        ));
    }
    let list: Vec<String> = chunk
        .steps
        .iter()
        .map(|s| {
            format!(
                "{}. {} [target: {}] [done when: {}]",
                s.step, s.action, s.target, s.success_criteria
            )
        })
        .collect();
    brief.push_str(&format!(
        "\nYOUR CHUNK (session {}, {} step(s)):\n{}\n\nWhen every step in the chunk is done, \
         reply with a single short line starting with DONE: and one sentence of evidence \
         (what you ran and what it showed). The planner reads that line to decide whether to \
         continue, so make it checkable, not a promise.",
        chunk.index + 1,
        chunk.steps.len(),
        list.join("\n")
    ));

    vec![
        Message::system(protocol::system_prompt(specs, &brief)),
        Message::user("Begin. Emit one tool call."),
    ]
}

/// Run one chunk to completion: ask, parse, execute tools, feed back, repeat.
///
/// Returns the executor's final `DONE:` line. `max_turns` bounds the loop; a
/// chunk that needs more turns than that is reported rather than silently
/// truncated, because the planner has to know the chunk did not finish.
pub fn run_chunk(
    client: &Client,
    seed: Vec<Message>,
    cwd: &Path,
    max_turns: u32,
    perm: crate::cli::PermissionMode,
) -> Result<String> {
    let mut history = seed;
    let mut reply = client.complete(&mut history)?;

    // Cheap models sometimes open with prose. Unlike the browser channel there is
    // no human to nudge, so ask once, mechanically, in the same shape.
    if matches!(protocol::parse_reply(&reply), Reply::Text(_)) {
        history.push(Message::user(
            "No tool call was emitted. Reply with ONLY one line of JSON of the shape \
             {\"tool_calls\":[{\"id\":\"call_0\",\"name\":\"list_dir\",\"input\":{\"path\":\".\"}}]} \
             and then continue the task with one such line per tool call.",
        ));
        reply = client.complete(&mut history)?;
    }

    for turn in 1..=max_turns {
        match protocol::parse_reply(&reply) {
            Reply::Text(final_line) => {
                let line = final_line.trim().to_string();
                if !line.to_uppercase().starts_with("DONE") {
                    eprintln!(
                        "[chunk] executor stopped without a DONE line; last words: {}",
                        line.chars().take(200).collect::<String>()
                    );
                }
                return Ok(line);
            }
            Reply::Tools(calls) => {
                for call in &calls {
                    eprintln!("[chunk turn {turn}] tool: {}", call.name);
                }
                let results: Vec<ToolResult> =
                    calls.iter().map(|c| tools_execute(c, cwd, perm)).collect();
                let observation = protocol::render_results(&results);
                history.push(Message::user(observation));
                reply = client.complete(&mut history)?;
            }
        }
    }

    bail!(
        "chunk exhausted its {max_turns}-turn budget without finishing; the planner must \
         decide whether to retry it with a smaller chunk"
    )
}

/// Execute one tool call. Auto-approved: a free-tier executor runs the plan the
/// human already approved, and `PermissionMode` still gates destructive and
/// network commands inside `tools::execute`.
fn tools_execute(call: &ToolCall, cwd: &Path, perm: crate::cli::PermissionMode) -> ToolResult {
    crate::tools::execute(call, cwd, true, perm)
}

/// Where a key file would live, for diagnostics and tests.
pub fn key_path() -> Option<PathBuf> {
    crate::util::home_dir().map(|h| h.join(".chatgpt-use").join("openrouter.key"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::delegation::PlanStep;

    fn step(n: u32) -> PlanStep {
        PlanStep {
            step: n,
            action: format!("do thing {n}"),
            target: "src/x.rs".into(),
            success_criteria: "tests pass".into(),
        }
    }

    #[test]
    fn chunking_preserves_order_and_respects_the_size() {
        let steps: Vec<_> = (1..=7).map(step).collect();
        let chunks = chunk_steps(&steps, 3);
        assert_eq!(chunks.len(), 3, "7 steps in 3s -> 3 chunks");
        assert_eq!(chunks[0].steps.len(), 3);
        assert_eq!(chunks[2].steps.len(), 1, "last chunk holds the remainder");
        let flat: Vec<u32> = chunks
            .iter()
            .flat_map(|c| c.steps.iter().map(|s| s.step))
            .collect();
        assert_eq!(flat, vec![1, 2, 3, 4, 5, 6, 7], "no step lost or reordered");
        assert_eq!(
            chunks.iter().map(|c| c.index).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
    }

    #[test]
    fn a_zero_sized_chunk_is_refused_rather_than_looping_forever() {
        assert!(chunk_steps(&[step(1)], 0).is_empty());
    }

    #[test]
    fn the_request_body_has_the_shape_openrouter_requires() {
        let c = Client {
            api_key: "test-key".into(),
            model: "some/model:free".into(),
            timeout_secs: 60,
        };
        let body = c.request_body(&[Message::system("s"), Message::user("u")]);
        assert_eq!(body["model"], "some/model:free");
        assert_eq!(body["messages"].as_array().unwrap().len(), 2);
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["temperature"], 0.2);
    }

    #[test]
    fn completions_are_read_from_both_known_shapes() {
        assert_eq!(
            parse_completion(r#"{"choices":[{"message":{"content":"hello"}}]}"#).unwrap(),
            "hello"
        );
        assert_eq!(
            parse_completion(r#"{"choices":[{"text":"legacy"}]}"#).unwrap(),
            "legacy"
        );
    }

    #[test]
    fn malformed_or_empty_responses_say_so_instead_of_panicking() {
        assert!(parse_completion("not json at all").is_err());
        assert!(parse_completion(r#"{"choices":[]}"#).is_err());
        assert!(parse_completion(r#"{"choices":[{"message":{}}]}"#).is_err());
    }

    #[test]
    fn the_chunk_brief_forbids_the_behaviours_that_waste_a_paid_plan() {
        let specs = vec![ToolSpec {
            name: "write_file".into(),
            description: "write".into(),
            input_schema: json!({"type":"object"}),
        }];
        let msgs = chunk_messages(
            &specs,
            "ship the flag",
            &["--json prints valid JSON".into()],
            &["no new deps".into()],
            &Chunk {
                index: 1,
                steps: vec![step(4), step(5)],
            },
            &["step 3 added the parser".into()],
        );
        let sys = &msgs[0].content;
        assert!(
            sys.contains("EXECUTOR"),
            "must frame the model as hands, not brain"
        );
        assert!(sys.contains("ship the flag"));
        assert!(sys.contains("--json prints valid JSON"));
        assert!(
            sys.contains("no new deps"),
            "the planner's prohibitions must survive"
        );
        assert!(
            sys.contains("step 3 added the parser"),
            "earlier work must be carried over"
        );
        assert!(sys.contains("do thing 4") && sys.contains("do thing 5"));
        assert!(
            !sys.contains("do thing 3"),
            "other chunks' steps must not leak in"
        );
        assert!(
            sys.contains("DONE"),
            "the completion signal the planner parses"
        );
    }

    #[test]
    fn a_missing_key_explains_how_to_get_one() {
        // Both sources must be empty for this to mean anything: `api_key()` reads
        // the env var AND ~/.chatgpt-use/openrouter.key, so on a configured
        // machine there is no "missing key" to assert about.
        //
        // It also must not use unwrap_err(): the success value is the key itself,
        // and a panic on it would print a secret into the test log.
        if std::env::var("OPENROUTER_API_KEY").is_ok() || key_path().is_some_and(|p| p.exists()) {
            eprintln!("an OpenRouter key is configured; nothing to assert");
            return;
        }
        match api_key() {
            Ok(_) => panic!("api_key() succeeded with no key configured"),
            Err(e) => {
                let msg = e.to_string();
                assert!(msg.contains("openrouter.ai/keys"), "{msg}");
                assert!(msg.contains("OPENROUTER_API_KEY"), "{msg}");
            }
        }
    }

    /// Real network, so it is `#[ignore]`d: the suite must stay offline, per
    /// AGENTS.md. Run with `cargo test -- --ignored`. OpenRouter is not
    /// chatgpt.com, so this costs no plan quota and no throttled requests -- it
    /// exists to prove the curl plumbing (body file, headers, refusal parsing)
    /// rather than to assert about it.
    #[test]
    #[ignore]
    fn a_bad_key_reaches_openrouter_and_comes_back_as_a_readable_refusal() {
        let c = Client {
            api_key: "sk-or-v1-definitely-not-a-real-key".into(),
            model: DEFAULT_MODEL.into(),
            timeout_secs: 60,
        };
        let mut history = vec![Message::user("say ok")];
        let err = c.complete(&mut history).unwrap_err().to_string();
        eprintln!("openrouter said: {err}");
        assert!(
            err.to_lowercase().contains("openrouter")
                || err.to_lowercase().contains("unauthorised")
                || err.to_lowercase().contains("authorized")
                || err.to_lowercase().contains("key"),
            "the refusal should name the cause, not the parser: {err}"
        );
    }

    #[test]
    fn a_paid_model_is_refused_before_anything_is_spent() {
        // The guard is about money, so both shapes of mistake matter: the suffix
        // missing entirely, and the suffix right but the case wrong.
        assert!(require_free("meta-llama/llama-3.3-70b-instruct").is_err());
        assert!(require_free("openai/gpt-4o").is_err());
        assert!(require_free("anthropic/claude-sonnet-4:free").is_ok());
        assert!(require_free("qwen/qwen3.8-27b:free").is_ok());
        let msg = require_free("openai/gpt-4o").unwrap_err().to_string();
        assert!(msg.contains("not a free-tier model"), "{msg}");
    }

    /// The load-bearing assumption of the whole two-tier design: a cheap model
    /// will emit the rigid one-line tool-call JSON. If it cannot, `delegate` does
    /// not work at any price, so this earns a real (free) call rather than an
    /// assumption.
    ///
    /// Uses `complete` only, never `run_chunk` -- that would execute whatever
    /// tools the model asked for and let a test touch the disk.
    #[test]
    #[ignore]
    fn a_free_model_actually_follows_the_tool_protocol() {
        let client = Client::new(None, 120).expect("a key and a free model");
        let specs = crate::tools::builtin_specs();
        let chunk = Chunk {
            index: 0,
            steps: vec![crate::delegation::PlanStep {
                step: 1,
                action: "list the files in the current directory".into(),
                target: ".".into(),
                success_criteria: "a listing is shown".into(),
            }],
        };
        let mut history = chunk_messages(&specs, "probe the executor path", &[], &[], &chunk, &[]);
        let reply = client.complete(&mut history).expect("one free completion");
        eprintln!("MODEL {}\nREPLY {}", client.model, reply);
        match protocol::parse_reply(&reply) {
            Reply::Tools(calls) => {
                assert!(!calls.is_empty(), "a tool-call reply with no calls in it");
                let names: Vec<String> = calls.iter().map(|c| c.name.clone()).collect();
                eprintln!("PARSED {:?} as the first turn", names);
            }
            Reply::Text(t) => panic!(
                "the free model answered in prose instead of calling a tool, so the text \
                 protocol it is being driven by does not hold at this tier. Reply was: {}",
                t.chars().take(400).collect::<String>()
            ),
        }
    }

    #[test]
    fn pick_usable_skips_models_that_do_not_answer() {
        // The real probe makes a completion; this one is a set. What is under test
        // is the selection order and the give-up behaviour, not the network.
        let usable = std::collections::HashSet::from(["b"]);
        let got = pick_usable(&["a", "b", "c"], |m| usable.contains(m));
        assert_eq!(got, Some("b"), "must take the first answering model, not the first listed");
        assert_eq!(pick_usable(&["a", "c"], |_| false), None, "none answering is None");
        assert_eq!(pick_usable(&[], |_| true), None, "an empty shortlist cannot be probed");
    }

    #[test]
    fn every_candidate_executor_model_is_free() {
        // The whole economic premise is that the executor costs nothing. If one
        // paid model slips into the shortlist, a "free" run quietly bills.
        assert!(!PREFERRED_FREE_MODELS.is_empty());
        for m in PREFERRED_FREE_MODELS {
            assert!(m.ends_with(":free"), "{m} is not a free-tier model");
        }
        assert_eq!(DEFAULT_MODEL, PREFERRED_FREE_MODELS[0]);
    }

    #[test]
    #[ignore]
    fn the_resolved_default_actually_exists_on_openrouter_right_now() {
        // The failure this guards against already happened twice: a hardcoded
        // default that had been retired from the free tier, and then a shortlist
        // of models that are LISTED but do not serve. Cheap and keyless for the
        // list; the probe costs a few output tokens of free quota.
        let key = api_key().expect("this probe needs a configured OpenRouter key");
        let catalogue = free_models().expect("OpenRouter's model list should be reachable");
        assert!(
            catalogue.len() > 5,
            "suspiciously short catalogue: {catalogue:?}"
        );
        let picked = resolve_model(&key);
        assert!(
            catalogue.contains(&picked),
            "resolved {picked} is not in the catalogue"
        );
        assert!(picked.ends_with(":free"));
        assert!(serves(&key, &picked), "the resolver picked {picked}, which does not answer");
        eprintln!("resolved executor model (verified serving): {picked}");
    }
}
