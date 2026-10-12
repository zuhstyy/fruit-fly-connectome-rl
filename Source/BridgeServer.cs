using System.Collections.Concurrent;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Channels;

namespace Celeste.Mod.FlyBrainCeleste;

// Only this file touches the socket. The game thread never waits for socket I/O.
public sealed class BridgeServer : IDisposable
{
    public readonly int Port =
        int.Parse(Environment.GetEnvironmentVariable("FLYBRAIN_PORT") ?? "28462");
    private readonly string token = Environment.GetEnvironmentVariable("FLYBRAIN_TOKEN") ?? "";
    public readonly ConcurrentQueue<Command> Commands = new();
    private readonly AutoResetEvent commandReady = new(false);
    private readonly CancellationTokenSource stop = new();
    private TcpListener? listener;
    private TcpClient? client;
    private Channel<string>? replies;
    private int connection;
    public bool Connected => Volatile.Read(ref replies) != null;
    public int Connection => Volatile.Read(ref connection);
    public string Status { get; private set; } = "Starting bridge";

    public void Start()
    {
        listener = new TcpListener(IPAddress.Loopback, Port);
        try
        {
            listener.Start();
            Status = "Python disconnected - port " + Port;
            _ = Run();
        }
        catch (SocketException e)
        {
            Status = "Bridge cannot start: " + e.Message;
            throw new InvalidOperationException(Status, e);
        }
    }

    private async Task Run()
    {
        while (!stop.IsCancellationRequested)
        {
            try
            {
                using TcpClient next = await listener!.AcceptTcpClientAsync(stop.Token);
                client = next;
                next.NoDelay = true;
                int generation = Interlocked.Increment(ref connection);
                var outgoing = Channel.CreateBounded<string>(64);
                Volatile.Write(ref replies, outgoing);
                Status = "Python connected";
                Reply(new { type = "hello", protocol = 6, mod = "FlyBrainCeleste",
                            version = "4.1.0",
                            worker = Environment.GetEnvironmentVariable("FLYBRAIN_WORKER_ID") });
                using var connectionStop =
                    CancellationTokenSource.CreateLinkedTokenSource(stop.Token);
                using NetworkStream stream = next.GetStream();
                using var reader = new StreamReader(stream, new UTF8Encoding(false));
                using var writer =
                    new StreamWriter(stream, new UTF8Encoding(false)) { AutoFlush = true };
                Task sending = SendLines(writer, outgoing.Reader, connectionStop.Token);
                try
                {
                    while (!connectionStop.IsCancellationRequested)
                    {
                        string? line = await reader.ReadLineAsync(connectionStop.Token);
                        if (line == null)
                            break;
                        Command? command = null;
                        try
                        {
                            if (line.Length > 8192)
                                throw new JsonException("Command too large");
                            command = JsonSerializer.Deserialize<Command>(line);
                            if (command == null || command.Token != token || token.Length < 24 ||
                                !command.Valid())
                                throw new JsonException("Invalid command");
                            command.Connection = generation;
                            if (Commands.Count >= 8)
                                throw new JsonException("Send one command at a time");
                            Commands.Enqueue(command);
                            commandReady.Set();
                        }
                        catch (JsonException e)
                        {
                            Reply(new { type = "error", id = command?.Id ?? -1,
                                        message = e.Message });
                        }
                    }
                }
                finally
                {
                    connectionStop.Cancel();
                    outgoing.Writer.TryComplete();
                    next.Close();
                    try
                    {
                        await sending;
                    }
                    catch (Exception)
                    { /* socket already closed */
                    }
                }
            }
            catch (Exception e) when (e is IOException or SocketException or
                                          OperationCanceledException or ObjectDisposedException)
            {
                if (!stop.IsCancellationRequested)
                    Status = "Python disconnected";
            }
            finally
            {
                Volatile.Write(ref replies, null);
                client = null;
                while (Commands.TryDequeue(out _))
                {
                }
                if (!stop.IsCancellationRequested)
                    Status = "Python disconnected - port " + Port;
            }
        }
    }

    private static async Task SendLines(StreamWriter writer, ChannelReader<string> channel,
                                        CancellationToken token)
    {
        await foreach (string line in channel.ReadAllAsync(token))
            await writer.WriteLineAsync(line.AsMemory(), token);
    }

    public void Reply(object value)
    {
        Channel<string>? outgoing = Volatile.Read(ref replies);
        if (outgoing != null && !outgoing.Writer.TryWrite(JsonSerializer.Serialize(value)))
            client?.Close(); // A stalled controller must not leave the game under bot control.
    }

    // Sleep only while waiting for a new decision, waking immediately on input.
    public void WaitForCommand(int milliseconds = 2)
    {
        if (Commands.IsEmpty && Connected)
            commandReady.WaitOne(milliseconds);
    }

    public void Dispose()
    {
        stop.Cancel();
        listener?.Stop();
        client?.Close();
        Volatile.Read(ref replies)?.Writer.TryComplete();
    }
}

public sealed class Command
{
    [JsonPropertyName("token")]
    public string Token { get; set; } = "";
    [JsonPropertyName("chapter")]
    public int Chapter { get; set; }
    [JsonPropertyName("force")]
    public bool Force { get; set; }
    [JsonPropertyName("interact")]
    public bool Interact { get; set; }
    [JsonPropertyName("id")]
    public int Id { get; set; }
    [JsonPropertyName("type")]
    public string Type { get; set; } = "";
    [JsonPropertyName("move_x")]
    public int MoveX { get; set; }
    [JsonPropertyName("move_y")]
    public int MoveY { get; set; }
    [JsonPropertyName("jump")]
    public bool Jump { get; set; }
    [JsonPropertyName("dash")]
    public bool Dash { get; set; }
    [JsonPropertyName("grab")]
    public bool Grab { get; set; }
    [JsonPropertyName("frames")]
    public int Frames { get; set; } = 4;
    [JsonPropertyName("reward_total")]
    public double RewardTotal { get; set; }
    [JsonPropertyName("reward_last")]
    public double RewardLast { get; set; }
    [JsonPropertyName("reward_distance")]
    public double RewardDistance { get; set; }
    [JsonPropertyName("reward_ready")]
    public bool RewardReady { get; set; }
    [JsonPropertyName("reward_score")]
    public double RewardScore { get; set; }
    [JsonPropertyName("reward_event")]
    public string RewardEvent { get; set; } = "";
    [JsonPropertyName("brain_updates")]
    public int BrainUpdates { get; set; }
    [JsonPropertyName("learned_edges")]
    public int LearnedEdges { get; set; }
    [JsonPropertyName("brain_change")]
    public double BrainChange { get; set; }
    [JsonPropertyName("exploration")]
    public double Exploration { get; set; } = 1;
    [JsonPropertyName("slot")]
    public int Slot { get; set; }
    [JsonPropertyName("enabled")]
    public bool Enabled { get; set; }
    [JsonPropertyName("objective")]
    public string Objective { get; set; } = "Waiting";
    [JsonPropertyName("training_mode")]
    public string TrainingMode { get; set; } = "Learner";
    [JsonPropertyName("navigation_distance")]
    public double NavigationDistance { get; set; }
    [JsonPropertyName("room_clears")]
    public int RoomClears { get; set; }
    [JsonPropertyName("practice_resets")]
    public int PracticeResets { get; set; }
    [JsonPropertyName("teacher_trials")]
    public int TeacherTrials { get; set; }
    [JsonPropertyName("teacher_trial_deaths")]
    public int TeacherDeaths { get; set; }
    [JsonIgnore]
    public int Connection { get; set; }
    public bool Valid() =>
        Id >= 0 &&
        (Type is "status" or "step" or "start" or "retry" or "release" or "quit" or
                 "feedback" or "capture" or "restore" or "practice" or "clear") &&
        Slot >= 0 && Slot < 8 && Objective != null
        && Objective.Length <= 160 && TrainingMode != null && TrainingMode.Length <= 32 &&
        double.IsFinite(NavigationDistance) && NavigationDistance >= 0 && RoomClears >= 0 &&
        PracticeResets >= 0 && TeacherTrials >= 0 && TeacherDeaths >= 0 && BrainUpdates >= 0 &&
        LearnedEdges >= 0 && LearnedEdges <= 3326 &&
        double.IsFinite(BrainChange) && BrainChange >= 0 &&
        double.IsFinite(Exploration) && Exploration >= 1 && Exploration <= 3 &&
        double.IsFinite(RewardScore) && RewardEvent != null && RewardEvent.Length <= 100 &&
        double.IsFinite(RewardTotal) && double.IsFinite(RewardLast) &&
        double.IsFinite(RewardDistance) && Chapter >= 0 && Chapter <= 10 && MoveX >= -1 &&
        MoveX <= 1 && MoveY >= -1 && MoveY <= 1 && Frames >= 1 && Frames <= 30;
}
