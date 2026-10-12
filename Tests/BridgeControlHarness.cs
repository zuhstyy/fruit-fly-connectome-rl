// Integration checks for the real C# bridge and shared controls. No game stubs.
using System.Diagnostics;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;
using Celeste.Mod.FlyBrainCeleste;

string root = Path.GetFullPath(args[0]);
Directory.CreateDirectory(root);
string pause = Path.Combine(root, "PAUSE"), stop = Path.Combine(root, "STOP");
File.Delete(pause);
File.Delete(stop);
Environment.SetEnvironmentVariable("FLYBRAIN_PAUSE_FILE", pause);
Environment.SetEnvironmentVariable("FLYBRAIN_STOP_FILE", stop);
var controls = new WorkerControl();
void Check(bool valid, string why)
{
    if (!valid)
        throw new Exception(why);
}
Check(!controls.Paused && !controls.Stopped, "initial control state");
controls.SetPaused(true);
Check(controls.Paused && File.Exists(pause), "OFF must be shared with launcher");
controls.SetPaused(false);
Check(!controls.Paused, "ON must clear pause");
controls.Stop();
Check(controls.Paused && controls.Stopped, "F10 sets both markers");
File.Delete(pause);
File.Delete(stop);

using var probe = new TcpListener(IPAddress.Loopback, 0);
probe.Start();
int port = ((IPEndPoint)probe.LocalEndpoint).Port;
probe.Stop();
Environment.SetEnvironmentVariable("FLYBRAIN_PORT", port.ToString());
Environment.SetEnvironmentVariable("FLYBRAIN_TOKEN", new string('a', 64));
using var server = new BridgeServer();
server.Start();
using var client = new TcpClient();
await client.ConnectAsync(IPAddress.Loopback, port);
using var stream = client.GetStream();
using var reader = new StreamReader(stream, new UTF8Encoding(false));
using var writer = new StreamWriter(stream, new UTF8Encoding(false)) { AutoFlush = true };
async Task<JsonElement> Read()
{
    string? line = await reader.ReadLineAsync().WaitAsync(TimeSpan.FromSeconds(5));
    return JsonDocument.Parse(line ?? throw new Exception("Unexpected disconnect")).RootElement;
}
var hello = await Read();
Check(hello.GetProperty("protocol").GetInt32() == 6 &&
          hello.GetProperty("version").GetString() == "4.1.0",
      "handshake");
await writer.WriteLineAsync("{\"id\":1,\"type\":\"status\",\"token\":\"wrong\"}");
var denied = await Read();
Check(denied.GetProperty("type").GetString() == "error", "bad token rejection");
await writer.WriteLineAsync(JsonSerializer.Serialize(
    new { id = 2, type = "step", token = new string('a', 64), move_x = 1, frames = 4 }));
Command? cmd = null;
var timer = Stopwatch.StartNew();
while (!server.Commands.TryDequeue(out cmd) && timer.Elapsed.TotalSeconds < 5)
    await Task.Delay(5);
Check(cmd?.MoveX == 1 && cmd.Frames == 4, "action delivered to game queue");
server.Reply(new { type = "error", id = cmd!.Id, code = "control_paused", message = "OFF" });
var paused = await Read();
Check(paused.GetProperty("id").GetInt32() == 2 &&
          paused.GetProperty("code").GetString() == "control_paused",
      "pause response");
// Measure the real socket/queue path without Celeste's frame scheduling.
const int requests = 200;
var service = Task.Run(() =>
                       {
                           for (int i = 0; i < requests; i++)
                           {
                               var timeout = Stopwatch.StartNew();
                               Command? received;
                               while (!server.Commands.TryDequeue(out received))
                               {
                                   if (timeout.Elapsed.TotalSeconds > 5)
                                       throw new Exception("Command signal timeout");
                                   server.WaitForCommand(2);
                               }
                               server.Reply(new { type = "status", id = received.Id });
                           }
                       });
var roundTrips = new List<double>();
for (int i = 0; i < requests; i++)
{
    var started = Stopwatch.StartNew();
    await writer.WriteLineAsync(
        JsonSerializer.Serialize(new { id = i + 3, type = "status", token = new string('a', 64) }));
    var response = await Read();
    Check(response.GetProperty("id").GetInt32() == i + 3, "fast request correlation");
    roundTrips.Add(started.Elapsed.TotalMilliseconds);
}
await service;
roundTrips.Sort();
Console.WriteLine(JsonSerializer.Serialize(new { test = "raw C# localhost transport", requests,
                                                 p50_ms = roundTrips[requests / 2],
                                                 p95_ms = roundTrips[(int)(requests * .95)],
                                                 native_game = false }));
client.Close();
timer.Restart();
while (server.Connected && timer.Elapsed.TotalSeconds < 5)
    await Task.Delay(5);
Check(!server.Connected, "disconnect releases bridge state");
Console.WriteLine(
    "9 C# bridge/control checks and 200 correlated command-signal round trips passed.");
