using Celeste.Mod.FlyBrainCeleste;

int checks = 0;
void Check(bool ok, string message)
{
    if (!ok)
        throw new Exception(message);
    checks++;
}
var rooms = new List<RoomRoute.Room> { new("car", 0, 0, 320, 180), new("start", 320, 0, 640, 180),
                                       new("next", 640, 0, 960, 180), new("up", 640, -180, 960, 0),
                                       new("isolated", 2000, 0, 2320, 180) };
var route = new RoomRoute(rooms, "car");
Check(route.Rank("car") == 0 && route.Rank("start") == 1 && route.Rank("next") == 2,
      "room order comes from map");
Check(route.Forward("start", "next"), "next room remains forward");
Check(!route.Forward("start", "car"), "previous room cannot become a target");
Check(route.Forward("next", "up"), "upward routes supported");
Check(!route.Forward("next", "isolated") && route.Rank("isolated") == -1,
      "unreachable geometry has no forward reward");
rooms.Reverse();
var reversed = new RoomRoute(rooms, "car");
Check(reversed.Rank("next") == route.Rank("next"), "map enumeration does not set progression");
Check(!RoomRoute.Adjacent(new("a", 0, 0, 100, 100), new("b", 100, 100, 200, 200)),
      "corner touching is not an exit");
Check(!new RoomRoute(rooms, "missing").Forward("car", "start"),
      "unknown origin disables route rewards");
var intro = new List<RoomRoute.Room> { new("-1", 0, 0, 320, 180), new("0", 320, 0, 640, 180),
                                       new("1", 640, 0, 960, 180), new("2", 960, 0, 1280, 180) };
var corrected = new RoomRoute(intro, "0", true);
Check(corrected.Forward("0", "1") && !corrected.Forward("0", "-1"),
      "recorded bug: spawn 0 must not target car -1");
Check(corrected.Forward("-1", "0"), "escape the car room without toggling or restarting");
Check(corrected.Rank("-1") >= 0 && corrected.Rank("0") > corrected.Rank("-1"),
      "backtracking ranks agree with reward");
Console.WriteLine($"{checks} C# room-route checks passed (synthetic geometry, not live maps).");
