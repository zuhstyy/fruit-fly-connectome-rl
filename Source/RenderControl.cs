using System.Diagnostics;
using System.Text.Json;

namespace Celeste.Mod.FlyBrainCeleste;

// A hidden worker can skip drawing while retaining every normal physics step.
// SHOW GAMES updates this same preference file and drawing resumes automatically.
internal sealed class RenderControl
{
    private long checkedAt;
    private bool visible = true;
    private readonly string path =
        Environment.GetEnvironmentVariable("FLYBRAIN_VISIBILITY_FILE") ?? "";
    private readonly string worker =
        (int.Parse(Environment.GetEnvironmentVariable("FLYBRAIN_WORKER_ID") ?? "0") + 1).ToString();
    public bool Visible()
    {
        if (path == "" || Stopwatch.GetElapsedTime(checkedAt).TotalSeconds < .5)
            return visible;
        checkedAt = Stopwatch.GetTimestamp();
        try
        {
            using JsonDocument document = JsonDocument.Parse(File.ReadAllText(path));
            JsonElement root = document.RootElement;
            visible = root.TryGetProperty("default", out var fallback) && fallback.GetBoolean();
            if (root.TryGetProperty("workers", out var rows) &&
                rows.TryGetProperty(worker, out var row))
                visible = row.GetBoolean();
        }
        catch (Exception error) when (error is IOException or JsonException)
        {
            // A brief file lock keeps the last known preference.
        }
        return visible;
    }
}
