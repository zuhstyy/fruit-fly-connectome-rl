"""Account for actual controlled game frames, not invented years of learning."""


def throughput_summary(frames, seconds, phases=None):
    phases = phases or {}
    rate = frames / max(seconds, 0.001)
    return dict(
        controlled_frames=int(frames),
        wall_seconds=float(seconds),
        frames_per_second=rate,
        game_speed=rate / 60,
        aggregate_game_days=frames / (60 * 86400),
        projected_game_years_per_wall_day=rate / (60 * 365),
        largest_phase=max(phases, key=phases.get) if phases else "not measured",
        phase_seconds=phases,
        note="Aggregate policy-controlled simulation time across workers, not biological learning time or proof of skill.",
    )
