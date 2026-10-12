"""Read recorded game evidence without interrupting the learner. No fake clear claims."""

from pathlib import Path
from datetime import datetime, timezone
import html
import json
import os

ROOT = Path(__file__).resolve().parents[1]


def assess(records):
    if not records:
        return dict(
            verdict="Not enough evidence: no completed training updates recorded.",
            facts=[],
        )
    latest = records[-1]
    same = [r for r in records if r.get("run_id") == latest.get("run_id")]
    first = same[0]
    workers = latest.get("worker_rewards", [])
    completed = latest.get("completed_chapters", [])
    old_completed = set(
        latest.get("initial_completed_chapters", first.get("completed_chapters", []))
    )
    new_completed = sorted(set(completed) - old_completed)
    start_rooms = int(first.get("rooms_this_run", 0))
    rooms = int(latest.get("rooms_this_run", 0))
    old_rank = max(
        (w.get("best_route_rank", -1) for w in first.get("worker_rewards", [])),
        default=-1,
    )
    new_rank = max((w.get("best_route_rank", -1) for w in workers), default=-1)
    jumped = sum(w.get("jump_actions", 0) for w in workers)
    left_jump = sum(w.get("left_jump_actions", 0) for w in workers)
    backtracks = sum(w.get("backtracks", 0) for w in workers)
    if new_completed:
        verdict = "Confirmed game-reported chapter progress: " + ", ".join(
            map(str, new_completed)
        )
    elif len(same) >= 3 and new_rank > old_rank:
        verdict = "Observed forward room progress during this run; reliable room-solving is not yet established."
    elif len(same) >= 3:
        verdict = "No demonstrated forward room improvement since the first recorded update of this run."
    else:
        verdict = "Not enough evidence yet: collect more training updates."
    facts = [
        f"Updates recorded in this run: {len(same)}",
        f'Game-reported completed chapters (saved history): {completed or "none"}',
        f"Distinct rooms visited this run: {start_rooms} at first update -> {rooms} now (visits can include backtracking)",
        f"Highest route rank observed: {old_rank} -> {new_rank} (map heuristic outside Prologue)",
        f'Game deaths this run: {latest.get("deaths_this_run",0)}; backward room entries: {backtracks}',
        f"Left during jump: {left_jump}/{jumped} sampled jump actions"
        + (f" ({100*left_jump/jumped:.1f}%)" if jumped else ""),
        f'Recurrent connections changed: {latest.get("learned_edges",0)}/3326; model updates: {latest.get("brain_updates",0)}',
        f'Real game processes at saved update: {latest.get("live_game_processes",0)} / requested {latest.get("requested_workers","unknown (older version)")}',
        f'Collection rate: {latest.get("decisions_per_second",0):.1f} decisions/sec; last learning update: {latest.get("update_seconds",0):.3f} sec',
    ]
    curriculum = latest.get("curriculum", {})
    if curriculum:
        facts.extend(
            [
                f'Actual selected-exit clears: {curriculum.get("room_clears", 0)} / {curriculum.get("room_attempts", 0)} recorded training attempts (includes search assistance)',
                f'Frozen-policy room tests: {curriculum.get("evaluation_clears", 0)} / {curriculum.get("evaluation_attempts", 0)} cleared; room starts may differ',
                f'Success routes on disk: {latest.get("success_routes", 0)} from {latest.get("successful_rooms", 0)} rooms; imitation decisions: {latest.get("imitation_decisions_total", 0)}',
                f'Learner-only clears this run: {sum(w.get("learner_room_clears", 0) for w in workers)}; search-assisted clears: {sum(w.get("assisted_room_clears", 0) for w in workers)}',
                f'Native practice resets: {sum(w.get("practice_resets", 0) for w in workers)}; restore mismatches/errors: {sum(w.get("restore_failures", 0) for w in workers)}',
                f'Automatic search: {sum(w.get("teacher_actions", 0) for w in workers)} executed actions; {sum(w.get("teacher_trials", 0) for w in workers)} trials / {sum(w.get("teacher_trial_deaths", 0) for w in workers)} trial deaths (not PPO decisions)',
                "Backward map entries can be necessary puzzle navigation; they do not alone indicate regression.",
            ]
        )
    if len(same) >= 6:
        width = max(1, len(same) // 3)

        def totals(record):
            ws = record.get("worker_rewards", [])
            return sum(w.get("jump_actions", 0) for w in ws), sum(
                w.get("left_jump_actions", 0) for w in ws
            )

        early_j, early_l = totals(same[width - 1])
        base_j, base_l = totals(same[-width - 1])
        late_j, late_l = jumped - base_j, left_jump - base_l
        if early_j > 0 and late_j > 0:
            facts.append(
                f"Left-jump frequency, early -> recent window: {100*early_l/early_j:.1f}% -> {100*late_l/late_j:.1f}% (context-dependent, not a clear by itself)"
            )
    if latest.get("saved_unix"):
        facts.append(
            "Last saved update (UTC): "
            + datetime.fromtimestamp(latest["saved_unix"], timezone.utc).isoformat()
        )
    duration_counts = {}
    longest = {}
    for worker in workers:
        for key, value in worker.get("duration_counts", {}).items():
            duration_counts[key] = duration_counts.get(key, 0) + value
        for key, value in worker.get("longest_holds", {}).items():
            longest[key] = max(longest.get(key, 0), value)
    if duration_counts:
        facts.append(
            "Learned hold selections (frames: count): "
            + ", ".join(
                f"{k}: {v}"
                for k, v in sorted(
                    duration_counts.items(), key=lambda item: int(item[0])
                )
            )
        )
    if longest:
        facts.append(
            "Longest uninterrupted holds (simulation seconds): "
            + ", ".join(f"{k}: {v/60:.2f}s" for k, v in longest.items())
        )
    if latest.get("capacity_message"):
        facts.append("Worker shortfall: " + latest["capacity_message"])
    return dict(
        verdict=verdict,
        facts=facts,
        workers=workers,
        caveat="This compares recorded training observations. It is not a controlled before/after playtest. Changed weights, higher points, and visited rooms alone do not prove better gameplay.",
    )


def main(root=ROOT):
    root = Path(root)
    saved = root / "training"
    saved.mkdir(exist_ok=True)
    records = []
    path = saved / "progress.jsonl"
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                value = json.loads(line)
                if isinstance(value, dict):
                    records.append(value)
            except json.JSONDecodeError:
                pass  # A concurrent writer may have an unfinished last line.
    result = assess(records)
    from runtime import RunLock

    try:
        with RunLock(root):
            pass
        result["learner_status"] = "No active learner holds this package's run lock."
    except RuntimeError:
        result["learner_status"] = "Learner is running (run lock held)."
    from runtime import read_json
    import time

    live = read_json(saved / "live.json")
    evaluation = read_json(saved / "evaluation.json")
    performance = read_json(saved / "performance.json")
    if performance:
        result["performance"] = performance
        result["facts"].append(
            f"Last measured speed: {performance.get('frames_per_second',0):.0f} controlled frames/sec; largest phase: {performance.get('largest_phase','unknown')}. Aggregate game time is not evidence of skill."
        )
    if evaluation:
        result["evaluation"] = evaluation
        result["facts"].append(
            "Latest full-story frozen-policy test: "
            + (
                "game-reported Prologue-through-Summit clear"
                if evaluation.get("full_main_story_clear")
                else "full clear not established"
            )
            + f'; completed {evaluation.get("completed_chapters", [])}; checkpoint SHA256 {evaluation.get("checkpoint_sha256", "not recorded")}'
        )
    if live:
        active = (
            "Learner is running" in result["learner_status"]
            and time.time() - live.get("heartbeat", 0) < 10
        )
        result["learner_status"] += (
            f'\nCurrent live status: {live.get("phase")}; '
            + str(live.get("live_game_processes", 0) if active else 0)
            + f' / {live.get("requested_workers",0)} real games alive.'
        )
        result["learner_status"] += "\n" + (
            live.get("capacity_message") or live.get("message", "")
        )
    console = saved / "run.log"
    if console.exists():
        with console.open("rb") as f:
            f.seek(max(0, console.stat().st_size - 6000))
            result["recent_console"] = "\n".join(
                f.read().decode("utf-8", errors="replace").splitlines()[-20:]
            )
    text = (
        result["verdict"]
        + "\n\n"
        + "\n".join(result["facts"])
        + "\n\n"
        + result.get("caveat", "")
    )
    text += "\n\n" + result["learner_status"]
    for w in result.get("workers", []):
        text += f'\nWorker {w.get("worker")}: room {w.get("room")}, target {w.get("exit_room")}, points {w.get("score",0):.2f}, {w.get("event","")}'
    if result.get("recent_console"):
        text += (
            "\n\nRecent console output (may include earlier runs):\n"
            + result["recent_console"]
        )
    (saved / "verification.txt").write_text(text + "\n", encoding="utf-8")
    (saved / "verification.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    page = saved / "verification.html"
    page.write_text(
        '<!doctype html><meta charset="utf-8"><title>Fly Brain verification</title><style>body{background:#101725;color:#eef2fa;font:18px/1.6 system-ui;max-width:1000px;margin:50px auto;padding:24px}pre{white-space:pre-wrap}h1{color:#85dab5}</style><h1>Fly Brain — recorded evidence</h1><pre>'
        + html.escape(text)
        + "</pre>",
        encoding="utf-8",
    )
    print(text, flush=True)
    if os.name == "nt":
        os.startfile(page)
    return result


if __name__ == "__main__":
    main()
