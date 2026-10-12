namespace Celeste.Mod.FlyBrainCeleste;

internal sealed class WorkerControl
{
    private readonly string pauseFile, stopFile;
    private long checkedAt;
    private bool paused, stopped;
    public WorkerControl()
    {
        pauseFile = RequiredPath("FLYBRAIN_PAUSE_FILE");
        stopFile = RequiredPath("FLYBRAIN_STOP_FILE");
    }
    private static string RequiredPath(string name)
    {
        string path = Environment.GetEnvironmentVariable(name) ?? "";
        if (!Path.IsPathFullyQualified(path))
            throw new InvalidOperationException(name + " must be an absolute launcher path.");
        return path;
    }
    private void Refresh()
    {
        if (System.Diagnostics.Stopwatch.GetElapsedTime(checkedAt).TotalMilliseconds < 25)
            return;
        checkedAt = System.Diagnostics.Stopwatch.GetTimestamp();
        paused = File.Exists(pauseFile);
        stopped = File.Exists(stopFile);
    }
    public bool Paused
    {
        get {
            Refresh();
            return paused;
        }
    }
    public bool Stopped
    {
        get {
            Refresh();
            return stopped;
        }
    }
    public void SetPaused(bool paused)
    {
        if (paused)
            File.WriteAllText(pauseFile, "Bot control OFF\n");
        else
            File.Delete(pauseFile);
        this.paused = paused;
        checkedAt = 0;
    }
    public void Stop()
    {
        SetPaused(true);
        File.WriteAllText(stopFile, "Stopped from Celeste\n");
        stopped = true;
    }
}
