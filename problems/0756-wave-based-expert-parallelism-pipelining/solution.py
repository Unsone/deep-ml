def wave_pipeline_makespan(transfer_times, compute_times, send_times):
    """
    Compute the makespan of the three-stage wave pipeline.

    Args:
        transfer_times: list of per-wave token transfer durations
        compute_times: list of per-wave expert compute durations
        send_times: list of per-wave result send durations

    Returns:
        Total makespan (number).
    """
    t_end = c_end = s_end = 0
    for tr, cp, sd in zip(transfer_times, compute_times, send_times):
        t_end = t_end + tr
        c_end = max(t_end, c_end) + cp
        s_end = max(c_end, s_end) + sd
    return s_end
    pass