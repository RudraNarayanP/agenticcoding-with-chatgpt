//! Two-tier delegation: ChatGPT is the planner and reviewer, a free-tier
//! OpenRouter model is the hands.
//!
//! Why the split, in one line each:
//!
//! - A reasoning model's value is *deciding what should change*. It is wasted on
//!   emitting a 300-line file one token at a time, and that is the part that
//!   burns the quota.
//! - A free model is perfectly adequate at the typing, provided it is never asked
//!   to design anything. So it receives a chunk of an already-agreed plan and a
//!   set of local tools, and nothing else.
//! - The planner is also the only party that can say "that chunk went wrong." So
//!   after each chunk its verdict goes back into the SAME ChatGPT conversation --
//!   one request, not the ~45 a fresh page load costs -- and it can redirect, or
//!   stop the run, rather than discover at the end that step 2 was broken.
//!
//! `run` mode (Mode 2) asks ChatGPT to do the typing too. This command is the
//! answer for anyone who objects to that on cost grounds.

use crate::channel::{Channel, ChannelOptions};
use crate::cli::DelegateArgs;
use crate::delegation::{self, DelegationPacket, Mode, PlanStep, Verdict};
use crate::openrouter;
use crate::tools;
use anyhow::{bail, Context, Result};
use std::path::PathBuf;

/// What the planner's check-in reply amounts to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Directive {
    Proceed,
    /// The planner explicitly said to stop.
    Halt,
}

pub fn run(args: &DelegateArgs) -> Result<()> {
    let cwd: PathBuf = match &args.cwd {
        Some(dir) => PathBuf::from(dir),
        None => std::env::current_dir()?,
    };
    let context = read_context(&args.files)?;

    // Build the executor before opening the browser. Connecting costs ~45
    // chatgpt.com requests against an account that throttles by request count,
    // and a missing OpenRouter key is otherwise discovered only after paying for
    // the planning turn.
    let client = if args.dry_run {
        None
    } else {
        Some(openrouter::Client::new(
            args.exec_model.clone(),
            args.exec_timeout,
        )?)
    };

    // ---- tier 1: the planner ------------------------------------------------
    let opts = ChannelOptions {
        profile: args.channel.profile.clone(),
        session: args.channel.session.clone(),
        project: args.channel.project.clone(),
        timeout_secs: args.channel.timeout,
        model: args.channel.model.clone(),
        busy_fail: args.channel.busy == crate::cli::BusyPolicy::Fail,
        receipt: None,
    };
    let mut channel = Channel::connect(&opts)?;

    eprintln!("[plan] asking ChatGPT for a plan");
    let plan_reply = channel.send(&delegation::build_prompt(Mode::Plan, &args.task, &context));
    let plan_reply = match plan_reply {
        Ok(r) => r,
        Err(e) => {
            channel.close();
            return Err(e).context("planning turn failed");
        }
    };
    let packet = match delegation::parse_packet(&plan_reply) {
        Ok(p) => p,
        Err(e) => {
            channel.close();
            return Err(e).context("ChatGPT did not return a valid delegation packet");
        }
    };

    print!("{}\n", plan_summary(&packet, args.chunk_steps));
    crate::ledger::record(
        "delegate",
        serde_json::json!({
            "goal": packet.goal,
            "verdict": format!("{:?}", packet.verdict),
            "steps": packet.plan.len(),
            "chunk_steps": args.chunk_steps,
            "model": args.channel.model,
            "dry_run": args.dry_run,
        }),
    );

    if packet.verdict != Verdict::Proceed {
        channel.close();
        bail!(
            "planner verdict is {:?} -- refusing to execute.\n{:?}",
            packet.verdict,
            if packet.risks.is_empty() {
                vec!["(no risks listed)".to_string()]
            } else {
                packet.risks.clone()
            }
        );
    }
    if args.dry_run {
        channel.close();
        eprintln!("[dry-run] stopped before executing anything");
        return Ok(());
    }

    // ---- tier 2: the hands --------------------------------------------------
    let client = client.expect("built above, and only skipped for --dry-run");
    let specs = tools::builtin_specs();
    let chunks = openrouter::chunk_steps(&packet.plan, args.chunk_steps);
    eprintln!(
        "[execute] {} step(s) in {} chunk(s) on {} (planner: {:?})",
        packet.plan.len(),
        chunks.len(),
        client.model,
        args.channel
            .model
            .clone()
            .unwrap_or_else(|| "account default".into()),
    );

    let mut completed: Vec<String> = Vec::new();
    let mut failed = 0usize;

    for chunk in chunks {
        let span = step_span(&chunk.steps);
        eprintln!("[chunk {}] steps {span}", chunk.index + 1);
        let seed = openrouter::chunk_messages(
            &specs,
            &packet.goal,
            &packet.acceptance,
            &packet.do_not_do,
            &chunk,
            &completed,
        );
        let outcome = openrouter::run_chunk(
            &client,
            seed,
            &cwd,
            args.executor_turns,
            args.permission_mode,
        );
        let line = match outcome {
            Ok(done) => done,
            Err(e) => {
                failed += 1;
                format!("FAILED: {e:#}")
            }
        };
        completed.push(format!("chunk {} (steps {span}): {line}", chunk.index + 1));

        // The planner only ever sees a compact report, never the tool chatter --
        // that is what keeps this check-in to a single cheap turn.
        if !args.no_review {
            match channel.send(&check_in_prompt(&packet.goal, completed.last().unwrap())) {
                Ok(reply) => {
                    eprintln!("[planner] {}", one_line(&reply));
                    if parse_directive(&reply) == Directive::Halt {
                        eprintln!("[planner] said to stop; not running further chunks");
                        break;
                    }
                }
                Err(e) => {
                    // A failed check-in must not discard a chunk that already
                    // touched the disk: report it and keep going.
                    eprintln!("[planner] check-in failed ({e:#}); continuing without a verdict");
                }
            }
        }
    }

    // ---- closing verdict ----------------------------------------------------
    if !args.no_review {
        eprintln!("[review] asking the planner to judge the result");
        match channel.send(&final_report(&packet, &completed, failed)) {
            Ok(reply) => println!("{}", reply.trim()),
            Err(e) => eprintln!(
                "[review] final review failed ({e:#}); reporting the executor log instead"
            ),
        }
    } else {
        println!("{}", completed.join("\n"));
    }
    channel.close();
    Ok(())
}

/// `--file` contents as one context block, matching `ask`'s shape so the planner
/// prompt is identical between the two commands.
fn read_context(files: &[String]) -> Result<String> {
    let mut context = String::new();
    for path in files {
        let contents = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read context file: {path}"))?;
        context.push_str(&format!("### File: {path}\n```\n{contents}\n```\n\n"));
    }
    Ok(context)
}

/// "3-5", or "3" for a single step, for logs and reports.
pub fn step_span(steps: &[PlanStep]) -> String {
    match (steps.first(), steps.last()) {
        (Some(a), Some(b)) if a.step == b.step => format!("{}", a.step),
        (Some(a), Some(b)) => format!("{}-{}", a.step, b.step),
        _ => "(none)".into(),
    }
}

/// The plan, as the human sees it before anything runs.
pub fn plan_summary(packet: &DelegationPacket, chunk_steps: usize) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "GOAL   {}\nVERDICT {:?}\n",
        packet.goal, packet.verdict
    ));
    if !packet.summary.is_empty() {
        out.push_str(&format!("SUMMARY\n- {}\n", packet.summary.join("\n- ")));
    }
    out.push_str("PLAN\n");
    for s in &packet.plan {
        out.push_str(&format!(
            "  {}. {} [{}] — done when: {}\n",
            s.step, s.action, s.target, s.success_criteria
        ));
    }
    if !packet.do_not_do.is_empty() {
        out.push_str(&format!("DO NOT DO\n- {}\n", packet.do_not_do.join("\n- ")));
    }
    let chunks = chunk_num(packet.plan.len(), chunk_steps);
    out.push_str(&format!(
        "EXECUTION\n  {} step(s) -> {} chunk(s) of at most {}\n",
        packet.plan.len(),
        chunks,
        chunk_steps
    ));
    out
}

/// How many chunks N steps make at `size` each. Pure, so the arithmetic in the
/// summary above is actually checked.
pub fn chunk_num(steps: usize, size: usize) -> usize {
    if size == 0 || steps == 0 {
        return 0;
    }
    steps.div_ceil(size)
}

/// Ask for CONTINUE or STOP. Kept to one short line in, one short line out: this
/// turn exists to catch a bad chunk, not to re-plan.
pub fn check_in_prompt(goal: &str, last_outcome: &str) -> String {
    format!(
        "EXECUTOR REPORT for the task \"{goal}\". Latest chunk result:\n{last_outcome}\n\
         Reply with exactly one word, CONTINUE or STOP. STOP if that result shows the plan is \
         wrong, the file was damaged, or the criterion cannot be met; otherwise CONTINUE. \
         Do not restate the plan and do not add steps.",
    )
}

/// The closing report, asking for a review verdict.
pub fn final_report(packet: &DelegationPacket, completed: &[String], failed: usize) -> String {
    format!(
        "All chunks have run for the goal \"{goal}\".\n{log}\n\nFailures: {failed}.\n\
         Review the outcome against the acceptance criteria and reply with verdict: PROCEED or \
         REVISE, then 2-4 lines of what still needs doing.\nACCEPTANCE:\n- {acceptance}",
        goal = packet.goal,
        log = completed.join("\n"),
        acceptance = packet.acceptance.join("\n- "),
    )
}

/// Read a check-in reply. Anything other than an explicit stop is treated as
/// continue, and a halt must be unambiguous: an empty or garbled reply from the
/// planner should not silently abort work already in progress.
pub fn parse_directive(reply: &str) -> Directive {
    let up = reply.to_ascii_uppercase();
    if up.contains("STOP") || up.contains("HALT") || up.contains("ABORT") {
        Directive::Halt
    } else {
        Directive::Proceed
    }
}

/// Collapse a model reply to one stderr line.
fn one_line(s: &str) -> String {
    let t: String = s.split_whitespace().collect::<Vec<_>>().join(" ");
    t.chars().take(160).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::delegation::PlanStep;

    fn packet() -> DelegationPacket {
        DelegationPacket {
            goal: "add --json to status".into(),
            summary: vec!["one route".into()],
            plan: (1..=7)
                .map(|n| PlanStep {
                    step: n,
                    action: format!("step {n}"),
                    target: "src/cmd/status.rs".into(),
                    success_criteria: format!("criterion {n}"),
                })
                .collect(),
            risks: vec![],
            tests: vec![],
            acceptance: vec!["--json emits valid JSON".into()],
            do_not_do: vec!["no new deps".into()],
            verdict: Verdict::Proceed,
        }
    }

    #[test]
    fn chunk_arithmetic_matches_the_chunker() {
        assert_eq!(chunk_num(7, 3), 3);
        assert_eq!(chunk_num(6, 3), 2);
        assert_eq!(chunk_num(1, 3), 1);
        assert_eq!(chunk_num(0, 3), 0);
        assert_eq!(
            chunk_num(5, 0),
            0,
            "a zero chunk size must not divide by zero"
        );
        // The summary's number must be the real chunker's number.
        assert_eq!(
            chunk_num(7, 3),
            openrouter::chunk_steps(&packet().plan, 3).len()
        );
    }

    #[test]
    fn the_summary_shows_the_plan_the_prohibitions_and_the_chunking() {
        let s = plan_summary(&packet(), 3);
        assert!(s.contains("add --json to status"));
        assert!(s.contains("7. step 7"));
        assert!(
            s.contains("no new deps"),
            "the planner's prohibitions must be visible"
        );
        assert!(s.contains("7 step(s) -> 3 chunk(s) of at most 3"));
    }

    #[test]
    fn step_spans_read_the_way_a_human_scans_a_log() {
        let p = packet();
        assert_eq!(step_span(&p.plan[..3]), "1-3");
        assert_eq!(step_span(&p.plan[6..]), "7");
        assert_eq!(step_span(&[]), "(none)");
    }

    #[test]
    fn a_check_in_asks_for_one_word_not_another_essay() {
        let p = check_in_prompt("goal", "chunk 2 (steps 4-6): DONE: cargo test passed");
        assert!(p.contains("CONTINUE or STOP"));
        assert!(
            p.contains("cargo test passed"),
            "the evidence must reach the planner"
        );
        assert!(
            p.contains("Do not restate the plan"),
            "this turn must stay cheap"
        );
        // Deliberately short: a check-in is billed against the account's quota.
        assert!(p.len() < 700, "check-in prompt grew to {} chars", p.len());
    }

    #[test]
    fn only_an_explicit_stop_halts_the_run() {
        assert_eq!(parse_directive("STOP"), Directive::Halt);
        assert_eq!(parse_directive("stop."), Directive::Halt);
        assert_eq!(parse_directive("CONTINUE"), Directive::Proceed);
        assert_eq!(
            parse_directive(""),
            Directive::Proceed,
            "garbage must not abort in-flight work"
        );
        assert_eq!(
            parse_directive("verdict: revise"),
            Directive::Proceed,
            "revising is not stopping"
        );
    }

    #[test]
    fn the_final_report_hands_over_every_chunk_and_the_criteria() {
        let r = final_report(
            &packet(),
            &[
                "chunk 1 (steps 1-3): DONE: builds".into(),
                "chunk 2 (steps 4-6): FAILED: boom".into(),
            ],
            1,
        );
        assert!(r.contains("chunk 1") && r.contains("chunk 2"));
        assert!(r.contains("Failures: 1"));
        assert!(
            r.contains("--json emits valid JSON"),
            "review is against acceptance, not vibes"
        );
        assert!(r.contains("PROCEED or REVISE"));
    }
}
