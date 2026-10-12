using System.Diagnostics;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Input;
using Monocle;

namespace Celeste.Mod.FlyBrainCeleste;

// Only the launcher opts an isolated worker into automatic control.
public sealed class FlyBrainModule : EverestModule
{
    private BridgeServer? server;
    private BotInputs? inputs;
    private WorkerControl? control;
    private Command? pending;
    private Session? session;
    private bool worker, active, loading, completed, retrying, emergencyStop;
    private bool previousF9, previousF10, frameConsumed;
    private int framesLeft, simulated, deaths, chapter, generation, cutsceneTicks;
    private int speed = 1;
    private long requestStarted;
    private string lastRoom = "", message = "Waiting for LEARN", error = "";
    private FlyHud? hud;
    private readonly Dictionary<string, int> roomVisits = new();
    private int replyDeathCount;
    private TimeSpan clock;
    private double rewardTotal, rewardLast, rewardDistance;
    private bool rewardReady;
    private string targetFrom = "", targetRoom = "";
    private Vector2 exitPoint;
    private float exitDistance;
    private RoomRoute? route;
    private readonly Dictionary<string, (string room, Vector2 point)> targets = new();
    private double rewardScore;
    private int brainUpdates, learnedEdges;
    private double brainChange, exploration = 1;
    private string lastAction = "waiting";
    private string rewardEvent = "Waiting";
    private readonly PracticeStates practice = new();
    private readonly WorldObservation observation = new();
    private bool practiceEnabled;
    private string objective = "Waiting", trainingMode = "Learner";
    private double navigationDistance;
    private int roomClears, practiceResets, teacherTrials, teacherDeaths;
    private readonly RenderControl rendering = new();
    private readonly SimulationPacing pacing = new();
    private readonly bool fastSimulation =
        Environment.GetEnvironmentVariable("FLYBRAIN_FAST") != "0";
    private int sentWorldRevision = -1;

    public override void Load()
    {
        worker = Environment.GetEnvironmentVariable("FLYBRAIN_WORKER") == "1";
        if (!worker)
            return;
        control = new WorkerControl();
        GameAccess.SavePath();
        speed = Math.Clamp(int.Parse(Environment.GetEnvironmentVariable("FLYBRAIN_TICKS") ?? "1"),
                           1, 64);
        server = new BridgeServer();
        server.Start();
        On.Monocle.Engine.Update += UpdateEngine;
        On.Monocle.Engine.Draw += DrawEngine;
        On.Monocle.ParticleSystem.Update += UpdateParticles;
        On.Celeste.Level.Update += UpdateLevel;
        On.Celeste.Level.Pause += PauseLevel;
        Everest.Events.Player.OnDie += OnDie;
        Everest.Events.Level.OnExit += OnExit;
    }
    public override void Initialize()
    {
        base.Initialize();
        if (worker)
        {
            GameAccess.ConfigureBoot();
            practice.Initialize(typeof(FlyHud));
        }
    }
    public override void Unload()
    {
        if (!worker)
            return;
        On.Monocle.Engine.Update -= UpdateEngine;
        On.Monocle.Engine.Draw -= DrawEngine;
        On.Monocle.ParticleSystem.Update -= UpdateParticles;
        On.Celeste.Level.Update -= UpdateLevel;
        On.Celeste.Level.Pause -= PauseLevel;
        Everest.Events.Player.OnDie -= OnDie;
        Everest.Events.Level.OnExit -= OnExit;
        inputs?.Dispose();
        pacing.SetTraining(Engine.Instance, false);
        practice.Clear();
        server?.Dispose();
    }
    private void OnDie(Player p)
    {
        if (active)
        {
            deaths++;
            Console.WriteLine(
                $"[FlyBrain] Game death #{deaths} room={session?.Level} pos={p.Position} state={p.StateMachine.State}");
        }
    }
    private void OnExit(Level level, LevelExit exit, LevelExit.Mode mode, Session s, HiresSnow snow)
    {
        if (active && ReferenceEquals(s, session) &&
            mode.ToString().StartsWith("Completed", StringComparison.Ordinal))
        {
            completed = true;
            message = "Chapter completed by the game";
        }
    }
    private void Release(string reason, string code = "game_error")
    {
        message = reason;
        if (code == "game_error")
            error = reason;
        if (pending != null)
            server!.Reply(new { type = "error", id = pending.Id, code, message = reason });
        pending = null;
        active = loading = retrying = false;
        inputs?.Reset();
        pacing.SetTraining(Engine.Instance, false);
    }
    private void PollControls(Engine engine)
    {
        KeyboardState keyboard = Keyboard.GetState();
        bool f9 = engine.IsActive && keyboard.IsKeyDown(Keys.F9);
        bool f10 = engine.IsActive && keyboard.IsKeyDown(Keys.F10);
        if (f9 && !previousF9 && !emergencyStop)
            control!.SetPaused(!control.Paused);
        if (f10 && !previousF10)
            control!.Stop();
        previousF9 = f9;
        previousF10 = f10;
        if (control!.Stopped)
        {
            emergencyStop = true;
            Release("Stopped. Keyboard/controller control restored.", "control_stopped");
        }
        else if (control.Paused)
        {
            Release("Bot OFF - manual control. F9 resumes from chapter start.", "control_paused");
        }
        else if (active && (!server!.Connected || server.Connection != generation))
        {
            Release("Controller disconnected - manual control restored.", "disconnected");
        }
    }
    private static bool ChangingScene()
    {
        object? next = GameAccess.Get(typeof(Engine), "NextScene");
        return next != null && !ReferenceEquals(next, Engine.Scene);
    }
    private void DrawEngine(On.Monocle.Engine.orig_Draw original, Engine engine, GameTime time)
    {
        if (!active || loading || control?.Paused == true || rendering.Visible())
            original(engine, time);
    }
    private void UpdateParticles(On.Monocle.ParticleSystem.orig_Update original,
                                 ParticleSystem particles)
    {
        // Only skip visual aging when no image is drawn. Emit calls still run,
        // preserving the game's random-number consumption (as in CelesteTAS).
        if (!active || !fastSimulation || loading || rendering.Visible())
            original(particles);
    }
    private void UpdateEngine(On.Monocle.Engine.orig_Update original, Engine engine,
                              GameTime realTime)
    {
        bool advanced = false;
        try
        {
            PollControls(engine);
            pacing.SetTraining(engine, active && fastSimulation && control?.Paused != true);
            if (active && !loading && pending == null)
                server!.WaitForCommand();
            if (Engine.Scene is Level visible)
                EnsureHud(visible);
            double timeout = pending?.Type == "start" ? 120 : 45;
            if (pending != null && Stopwatch.GetElapsedTime(requestStarted).TotalSeconds > timeout)
                Release("Timed out: " + DescribeScene());

            for (int tick = 0; tick < speed; tick++)
            {
                if (pending == null && server!.Commands.TryDequeue(out Command? cmd))
                {
                    if (cmd.Connection != server.Connection)
                        continue;
                    Handle(cmd);
                }
                if (active && Engine.Scene is Level level &&
                    ReferenceEquals(level.Session, session))
                {
                    EnsureHud(level);
                    if (level.Paused)
                    {
                        control!.SetPaused(true);
                        Release("Paused from game - bot OFF. F9 resumes from chapter start.",
                                "control_paused");
                        advanced = true;
                        original(engine, realTime);
                        break;
                    }
                    if (loading && Ready(level) && !ChangingScene())
                    {
                        loading = false;
                        lastRoom = level.Session.Level;
                        roomVisits[lastRoom] = 1;
                        message = "Bot ON - learning";
                        ReplyState(level);
                    }
                    if (retrying && Ready(level))
                    {
                        retrying = false;
                        inputs?.Reset();
                        ReplyState(level);
                    }
                    if (completed && pending != null)
                        ReplyState(level);
                    // Keep the final live level available for a practice branch
                    // restore. A real start request below still advances chapters.
                    if (completed && practiceEnabled && pending == null)
                        break;
                    // Never block queued scene changes. PollControls still runs on
                    // every outer frame while waiting for a policy decision.
                    if (!ChangingScene() && !loading && pending == null &&
                        (Ready(level) || completed))
                        break;
                    frameConsumed = false;
                    clock += TimeSpan.FromSeconds(1.0 / 60);
                    advanced = true;
                    original(engine, new GameTime(clock, TimeSpan.FromSeconds(1.0 / 60)));
                    simulated++;
                    if (pending?.Type == "step")
                    {
                        if (frameConsumed)
                            framesLeft--;
                        Player? player = level.Tracker.GetEntity<Player>();
                        bool changed = level.Session.Level != lastRoom;
                        if (changed)
                        {
                            lastRoom = level.Session.Level;
                            roomVisits[lastRoom] = roomVisits.GetValueOrDefault(lastRoom) + 1;
                        }
                        if (framesLeft <= 0 || deaths > replyDeathCount ||
                            (player != null && player.Dead) || changed || completed)
                            ReplyState(level);
                    }
                }
                else
                {
                    if (active && completed && pending != null)
                        ReplyState(null);
                    // Boot and all manual/menu play always receive normal input.
                    advanced = true;
                    original(engine, realTime);
                    break;
                }
            }
        }
        catch (Exception e)
        {
            Console.WriteLine("[FlyBrain] " + e);
            Release(e.GetType().Name + ": " + (e.InnerException?.Message ?? e.Message) + " | " +
                    DescribeScene());
        }
        if (!advanced && !active)
            original(engine, realTime);
        pacing.SetTraining(engine, active && fastSimulation && control?.Paused != true);
        pacing.PrepareDraw(engine, loading || rendering.Visible());
    }
    private void UpdateLevel(On.Celeste.Level.orig_Update original, Level level)
    {
        if (!active || control!.Paused || control.Stopped ||
            !ReferenceEquals(level.Session, session))
        {
            original(level);
            return;
        }
        // This hook is AFTER MInput.Update. Applying inputs around Engine.Update
        // allowed the engine's input refresh/disabled flag to interfere with them.
        inputs ??= new BotInputs();
        Command action = pending?.Type == "step" ? pending : new Command();
        frameConsumed = pending?.Type == "step" && Ready(level);
        if (GameAccess.Get(level, "InCutscene") is true)
        {
            cutsceneTicks++;
            if (cutsceneTicks % 60 == 0)
                GameAccess.Call(level, "SkipCutscene");
            action = new Command { Interact = cutsceneTicks % 12 == 0 };
        }
        else
            cutsceneTicks = 0;
        inputs.Run(action, () => original(level));
    }
    private void PauseLevel(On.Celeste.Level.orig_Pause original, Level level, int startIndex,
                            bool minimal, bool quickReset)
    {
        if (active && control!.Paused == false && !Engine.Instance.IsActive)
            return;
        original(level, startIndex, minimal, quickReset);
    }
    private static bool Ready(Level level)
    {
        Player? p = level.Tracker.GetEntity<Player>();
        // The Prologue's bird tutorial waits for dash while InControl is false.
        bool canAct =
            p != null && (p.InControl || p.StateMachine.State == Player.StBirdDashTutorial);
        return p != null && !p.Dead && canAct && !level.Paused &&
               GameAccess.Get(level, "Transitioning") is not true &&
               GameAccess.Get(level, "InCutscene") is not true;
    }
    private static string DescribeScene()
    {
        if (Engine.Scene is not Level l)
            return "scene=" + (Engine.Scene?.GetType().Name ?? "initializing");
        Player? p = l.Tracker.GetEntity<Player>();
        return $"scene=Level room={l.Session.Level} player={p != null} state={p?.StateMachine.State} paused={l.Paused} cutscene={GameAccess.Get(l, "InCutscene")} inputDisabled={MInput.Disabled}";
    }
    private void Handle(Command cmd)
    {
        if (cmd.RewardReady)
        {
            rewardReady = true;
            rewardTotal = cmd.RewardTotal;
            rewardLast = cmd.RewardLast;
            rewardDistance = cmd.RewardDistance;
            rewardScore = cmd.RewardScore;
            rewardEvent = cmd.RewardEvent;
            brainUpdates = cmd.BrainUpdates;
            learnedEdges = cmd.LearnedEdges;
            brainChange = cmd.BrainChange;
            exploration = cmd.Exploration;
            objective = cmd.Objective;
            trainingMode = cmd.TrainingMode;
            navigationDistance = cmd.NavigationDistance;
            roomClears = cmd.RoomClears;
            practiceResets = cmd.PracticeResets;
            teacherTrials = cmd.TeacherTrials;
            teacherDeaths = cmd.TeacherDeaths;
        }
        pending = cmd;
        requestStarted = Stopwatch.GetTimestamp();
        if (cmd.Type == "feedback")
        {
            server!.Reply(new { type = "feedback", id = cmd.Id });
            pending = null;
            return;
        }
        if (cmd.Type == "status")
        {
            server!.Reply(new { type = "status",
                                id = cmd.Id,
                                protocol = 6,
                                version = "4.1.0",
                                scene = Engine.Scene?.GetType().Name,
                                active,
                                loading,
                                completed,
                                control_paused = control!.Paused,
                                emergency_stop = emergencyStop,
                                message,
                                error,
                                chapter,
                                frame = simulated,
                                deaths,
                                room = session?.Level,
                                save_path = GameAccess.SavePath(),
                                worker_pid = Environment.ProcessId,
                                practice_available = practice.Available,
                                practice_enabled = practiceEnabled,
                                fast_simulation = pacing.Fast,
                                fixed_physics_hz = 60,
                                worker_path = Path.GetFullPath(AppContext.BaseDirectory) });
            pending = null;
            return;
        }
        if (cmd.Type == "practice")
        {
            practiceEnabled = cmd.Enabled;
            if (!practiceEnabled)
                practice.Clear();
            server!.Reply(new { type = "practice", id = cmd.Id, enabled = practiceEnabled,
                                available = practice.Available, message = practice.Error });
            pending = null;
            return;
        }
        if (cmd.Type == "quit")
        {
            active = loading = retrying = false;
            inputs?.Reset();
            server!.Reply(new { type = "quitting", id = cmd.Id });
            pending = null;
            Engine.Instance.Exit();
            return;
        }
        if (cmd.Type == "release")
        {
            active = loading = retrying = false;
            inputs?.Reset();
            message = "Bot OFF - manual control";
            server!.Reply(new { type = "released", id = cmd.Id });
            pending = null;
            return;
        }
        if (emergencyStop || control!.Stopped)
        {
            Release("Emergency stop. Start a new run in the launcher.", "control_stopped");
            return;
        }
        if (control.Paused)
        {
            Release("Bot control OFF", "control_paused");
            return;
        }
        framesLeft = cmd.Frames;
        if (cmd.Type == "clear")
        {
            practice.ClearSlot(cmd.Slot);
            server!.Reply(new { type = "cleared", id = cmd.Id, slot = cmd.Slot });
            pending = null;
            return;
        }
        if (cmd.Type is "capture" or "restore")
        {
            if (!active || !practiceEnabled || Engine.Scene is not Level stateLevel)
            {
                server!.Reply(new { type = "error", id = cmd.Id,
                                    message = "Practice is disabled or not in a level" });
                pending = null;
                return;
            }
            inputs ??= new BotInputs();
            if (cmd.Type == "restore" && ChangingScene())
                Engine.Scene = stateLevel; // Cancel a branch's queued scene exit before restoring.
            bool ok = cmd.Type == "capture"
                          ? Ready(stateLevel) && practice.Capture(cmd.Slot, inputs)
                          : practice.Restore(cmd.Slot, inputs);
            if (!ok)
            {
                server!.Reply(new { type = "error", id = cmd.Id,
                                    message = practice.Error == "" ? "Practice operation rejected"
                                                                   : practice.Error });
                pending = null;
                return;
            }
            if (cmd.Type == "restore")
            {
                session = stateLevel.Session;
                lastRoom = session.Level;
                completed = loading = retrying = false;
                observation.Reset();
                targetFrom = "";
                EnsureHud(stateLevel);
            }
            ReplyState(stateLevel);
            return;
        }
        if (cmd.Type == "step")
            lastAction =
                $"X={cmd.MoveX} Y={cmd.MoveY} jump={cmd.Jump} dash={cmd.Dash} grab={cmd.Grab} hold={cmd.Frames}f ({cmd.Frames / 60.0:0.00}s)";
        if (cmd.Type == "start")
        {
            if (completed && practiceEnabled && Engine.Scene is Level completedLevel &&
                ChangingScene())
                Engine.Scene = completedLevel;
            if (ChangingScene() ||
                Engine.Scene?.GetType().Name is not("Overworld" or "Level" or
                                                    "AreaComplete" or "LevelExit") ||
                AreaData.Areas == null || AreaData.Areas.Count <= cmd.Chapter ||
                Input.MoveX == null)
            {
                server!.Reply(
                    new { type = "error", id = cmd.Id, message = "not_ready: " + DescribeScene() });
                pending = null;
                return;
            }
            GameAccess.PrepareSave();
            GameAccess.ConfigureBackground();
            practice.Clear();
            observation.NewChapter();
            sentWorldRevision = -1;
            targetFrom = targetRoom = "";
            route = null;
            targets.Clear();
            rewardReady = false;
            rewardScore = rewardLast = rewardDistance = 0;
            chapter = cmd.Chapter;
            session = new Session(new AreaKey(chapter, AreaMode.Normal));
            GameAccess.EnterChapter(session);
            completed = false;
            active = loading = true;
            retrying = false;
            deaths = simulated = replyDeathCount = cutsceneTicks = 0;
            generation = server!.Connection;
            inputs?.Reset();
            roomVisits.Clear();
            error = "";
            message = "Opening training save / chapter " + chapter;
            Console.WriteLine("[FlyBrain] " + message);
        }
        else if (!active)
            Release("Worker is stopped; start a chapter first.");
        else if (cmd.Type == "retry")
        {
            if (Engine.Scene is not Level l)
            {
                Release("Cannot retry outside a level");
                return;
            }
            Player? p = l.Tracker.GetEntity<Player>();
            // Never manufacture deaths for a timer, a stalled policy, or a retry.
            // If already alive after Celeste respawned, return its state directly.
            if (Ready(l))
            {
                ReplyState(l);
                return;
            }
            retrying = true;
        }
    }
    private void EnsureHud(Level level)
    {
        if (hud != null && hud.Scene == level)
            return;
        hud = new FlyHud(this);
        level.Add(hud);
    }
    private Vector2 Target(Level l, Player? p)
    {
        Vector2 at = p?.Center ?? l.Bounds.Center.ToVector2();
        if (route == null)
        {
            var rooms = new List<RoomRoute.Room>();
            foreach (LevelData data in l.Session.MapData.Levels)
            {
                Rectangle b = data.Bounds;
                rooms.Add(new RoomRoute.Room(data.Name, b.Left, b.Top, b.Right, b.Bottom));
            }
            route = new RoomRoute(rooms, l.Session.MapData.StartLevel().Name, chapter == 0);
        }
        if (targetFrom != l.Session.Level)
        {
            targetFrom = l.Session.Level;
            targetRoom = "";
            exitPoint = at;
            float best = float.PositiveInfinity;
            bool cached = targets.TryGetValue(targetFrom, out var savedTarget);
            if (cached)
            {
                targetRoom = savedTarget.room;
                exitPoint = savedTarget.point;
            }
            Rectangle a = l.Bounds;
            foreach (LevelData room in l.Session.MapData.Levels)
            {
                if (cached || !route.Forward(l.Session.Level, room.Name))
                    continue;
                Rectangle b = room.Bounds;
                var candidates = new List<Vector2>();
                if (Math.Abs(b.Left - a.Right) <= 8 || Math.Abs(b.Right - a.Left) <= 8)
                {
                    float x = Math.Abs(b.Left - a.Right) <= 8 ? a.Right : a.Left;
                    for (int y = Math.Max(a.Top, b.Top) + 8; y < Math.Min(a.Bottom, b.Bottom) - 8;
                         y += 8)
                        candidates.Add(new Vector2(x, y));
                }
                if (Math.Abs(b.Top - a.Bottom) <= 8 || Math.Abs(b.Bottom - a.Top) <= 8)
                {
                    float y = Math.Abs(b.Top - a.Bottom) <= 8 ? a.Bottom : a.Top;
                    for (int x = Math.Max(a.Left, b.Left) + 8; x < Math.Min(a.Right, b.Right) - 8;
                         x += 8)
                        candidates.Add(new Vector2(x, y));
                }
                foreach (Vector2 candidate in candidates)
                {
                    Vector2 inside = new(Math.Clamp(candidate.X, a.Left + 8, a.Right - 8),
                                         Math.Clamp(candidate.Y, a.Top + 8, a.Bottom - 8));
                    bool blocked = false;
                    foreach (Vector2 offset in new[] { Vector2.Zero, new Vector2(-3, 0),
                                                       new Vector2(3, 0), new Vector2(0, -6),
                                                       new Vector2(0, 6) })
                        if (l.CollideCheck<Solid>(inside + offset) ||
                            l.CollideCheck<Spikes>(inside + offset))
                            blocked = true;
                    if (blocked)
                        continue;
                    float score = Vector2.Distance(at, candidate);
                    if (score < best)
                    {
                        best = score;
                        exitPoint = candidate;
                        targetRoom = room.Name;
                    }
                }
            }
        }
        if (targetRoom != "")
            targets[targetFrom] = (targetRoom, exitPoint);
        // Stable target throughout the room: no reward from switching nearest exits.
        // Boundary checks are a geometry hint, not a full pathfinding guarantee.
        exitDistance = targetRoom == "" ? 0 : Vector2.Distance(at, exitPoint);
        return targetRoom == "" ? at : exitPoint;
    }
    private void ReplyState(Level? l)
    {
        if (pending == null)
            return;
        Player? p = l?.Tracker.GetEntity<Player>();
        Vector2 pos = p?.Center ?? Vector2.Zero;
        bool navigationReady = l != null && Ready(l);
        Vector2 target = navigationReady ? Target(l!, p) : pos;
        WorldObservation.RoomGrid? map = navigationReady ? observation.Map(l!) : null;
        bool sendWorld = navigationReady && observation.Revision != sentWorldRevision;
        if (sendWorld)
            sentWorldRevision = observation.Revision;
        var objects = l == null ? new List<WorldObservation.ObjectInfo>()
                                : observation.Objects(l, pos, simulated);
        Vector2 wind =
            l != null && GameAccess.Get(l, "Wind") is Vector2 value ? value : Vector2.Zero;
        var grid = new float[81];
        var markers = new List<(Vector2 pos, float value)>();
        if (l != null)
            foreach (Entity e in l.Entities)
            {
                if (!e.Collidable)
                    continue;
                string name = e.GetType().Name;
                if (name.Contains("Spinner") || name.Contains("Seeker") || name.Contains("Lava") ||
                    name.Contains("FireBall"))
                    markers.Add((e.Center, -0.75f));
                else if (name.Contains("Refill") || name.Contains("Spring") ||
                         name.Contains("Booster"))
                    markers.Add((e.Center, 0.5f));
                else if (name.Contains("Key") || name.Contains("TheoCrystal") ||
                         name.Contains("Switch") || name.Contains("Door"))
                    markers.Add((e.Center, 0.25f));
            }
        if (l != null)
            for (int y = 0; y < 9; y++)
                for (int x = 0; x < 9; x++)
                {
                    Vector2 point = pos + new Vector2((x - 4) * 12, (y - 4) * 12);
                    float cell = l.CollideCheck<Spikes>(point)  ? -1
                                 : l.CollideCheck<Solid>(point) ? 1
                                                                : 0;
                    foreach (var marker in markers)
                        if (Vector2.DistanceSquared(point, marker.pos) < 144 && cell != -1)
                            cell = marker.value;
                    grid[y * 9 + x] = cell;
                }
        server!.Reply(new {
            type = "state",
            id = pending.Id,
            protocol = 6,
            chapter,
            room = session?.Level ?? "",
            x = pos.X,
            y = pos.Y,
            vx = p?.Speed.X ?? 0f,
            vy = p?.Speed.Y ?? 0f,
            grounded = p != null && !p.Dead && p.OnGround(),
            wall_left = p != null && !p.Dead && p.CollideCheck<Solid>(p.Position - Vector2.UnitX),
            wall_right = p != null && !p.Dead && p.CollideCheck<Solid>(p.Position + Vector2.UnitX),
            dashes = p?.Dashes ?? 0,
            stamina = p?.Stamina ?? 0f,
            player_state = p?.StateMachine.State ?? -1,
            in_control = p?.InControl ?? false,
            holding = p != null && GameAccess.Get(p, "Holding") != null,
            jump_held = ((inputs?.Capture() ?? 0) & ButtonHistory.Jump) != 0,
            dash_held = ((inputs?.Capture() ?? 0) & ButtonHistory.Dash) != 0,
            grab_held = ((inputs?.Capture() ?? 0) & ButtonHistory.Grab) != 0,
            wind_x = wind.X,
            wind_y = wind.Y,
            keys = session != null && GameAccess.Get(session, "Keys") is object keys &&
                           GameAccess.Get(keys, "Count") is int keyCount
                       ? keyCount
                       : 0,
            map = sendWorld ? map : null,
            world_revision = observation.Revision,
            objects,
            route_graph = sendWorld ? observation.Graph(l!) : null,
            portals = sendWorld ? observation.Portals(l!, pos) : null,
            practice_available = practice.Available,
            practice_enabled = practiceEnabled,
            restored = pending.Type == "restore",
            slot = pending.Slot,
            goal_dx = target.X - pos.X,
            goal_dy = target.Y - pos.Y,
            exit_id = !navigationReady || targetRoom == ""
                          ? ""
                          : targetFrom + ":" + targetRoom + ":" + exitPoint.X + ":" + exitPoint.Y,
            exit_room = targetRoom,
            exit_distance = exitDistance,
            route_rank = l == null ? -1 : route?.Rank(l.Session.Level) ?? -1,
            exit_forward = navigationReady && targetRoom != "" &&
                           route?.Forward(l!.Session.Level, targetRoom) == true,
            bounds_x = l?.Bounds.X ?? 0,
            bounds_y = l?.Bounds.Y ?? 0,
            bounds_width = l?.Bounds.Width ?? 1,
            bounds_height = l?.Bounds.Height ?? 1,
            grid,
            dead = !completed && (deaths > replyDeathCount || (p != null && p.Dead)),
            completed,
            deaths,
            action_frames = pending.Type == "step" ? Math.Max(0, pending.Frames - framesLeft) : 0,
            frame = simulated,
            ready = l != null && Ready(l),
            error
        });
        replyDeathCount = deaths;
        pending = null;
    }
    private sealed class FlyHud(FlyBrainModule owner) : Entity
    {
        public override void Added(Scene scene)
        {
            base.Added(scene);
            Tag = Tags.HUD | Tags.Persistent;
            Depth = -100000;
        }
        public override void Render()
        {
            Draw.Rect(8, 8, 1800, 320, Color.Black * 0.75f);
            bool on = owner.active && owner.control?.Paused == false;
            string mode = on ? "BOT ON" : "BOT OFF / MANUAL";
            ActiveFont.DrawOutline("Fly Brain 4.1 | " + mode + " | F9: ON/OFF | F10: stop all",
                                   new Vector2(20, 16), Vector2.Zero, Vector2.One * 0.55f,
                                   on ? Color.LightGreen : Color.Yellow, 2, Color.Black);
            string info =
                $"Chapter {owner.chapter} / {owner.session?.Level} | {owner.deaths} game deaths | {owner.lastAction}";
            ActiveFont.DrawOutline(info, new Vector2(20, 52), Vector2.Zero, Vector2.One * 0.45f,
                                   Color.White, 2, Color.Black);
            string points =
                owner.rewardReady
                    ? $"ATTEMPT POINTS {owner.rewardScore:+0.00;-0.00;0.00} | last {owner.rewardLast:+0.000;-0.000;0.000} | distance {owner.rewardDistance:+0.000;-0.000;0.000}"
                    : "POINTS: waiting for training (movement check is unscored)";
            ActiveFont.DrawOutline(points, new Vector2(20, 86), Vector2.Zero, Vector2.One * 0.48f,
                                   owner.rewardScore < 0 ? Color.OrangeRed : Color.LightGreen, 2,
                                   Color.Black);
            string destination =
                $"Objective: {owner.objective} | route distance {owner.navigationDistance:0} px (static hint)";
            ActiveFont.DrawOutline(destination, new Vector2(20, 124), Vector2.Zero,
                                   Vector2.One * 0.42f, Color.White, 2, Color.Black);
            ActiveFont.DrawOutline(
                $"{owner.rewardEvent} | Run return {owner.rewardTotal:+0.0;-0.0;0.0}",
                new Vector2(20, 166), Vector2.Zero, Vector2.One * 0.40f, Color.White, 2,
                Color.Black);
            string learning =
                $"FLY CONNECTIONS LEARNING | update {owner.brainUpdates} | {owner.learnedEdges}/3326 changed | exploration x{owner.exploration:0.0}";
            ActiveFont.DrawOutline(learning, new Vector2(20, 206), Vector2.Zero,
                                   Vector2.One * 0.40f, Color.Cyan, 2, Color.Black);
            string practiceInfo =
                $"{owner.trainingMode} | clears {owner.roomClears} | practice resets {owner.practiceResets} | search trials {owner.teacherTrials} / {owner.teacherDeaths} trial deaths";
            ActiveFont.DrawOutline(practiceInfo, new Vector2(20, 244), Vector2.Zero,
                                   Vector2.One * 0.35f, Color.White, 2, Color.Black);
            string status = owner.error != "" ? owner.error : owner.message;
            ActiveFont.DrawOutline(status, new Vector2(20, 282), Vector2.Zero, Vector2.One * 0.35f,
                                   owner.error != "" ? Color.OrangeRed : Color.White, 2,
                                   Color.Black);
        }
    }
}
