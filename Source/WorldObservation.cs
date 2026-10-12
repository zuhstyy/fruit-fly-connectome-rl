using Microsoft.Xna.Framework;
using Monocle;
using System.Linq;

namespace Celeste.Mod.FlyBrainCeleste;

// Static geometry is cached per room. Object positions are measured every reply.
// This is a collision hint for navigation, not a substitute Celeste simulator.
internal sealed class WorldObservation
{
    public sealed record ObjectInfo(string kind, string name, float x, float y, float vx, float vy,
                                    float width, float height);
    public sealed record RoomGrid(float x, float y, int cell, int width, int height, string tiles);
    public sealed record Portal(string room, float x, float y);
    public sealed record RoomInfo(string name, string[] neighbors, bool terminal_hint);
    private readonly Dictionary<Entity, (Vector2 pos, int frame)> previous = new();
    private RoomGrid? grid;
    private string room = "";
    private List<RoomInfo>? graph;
    public int Revision { get; private set; }

    public void Reset()
    {
        grid = null;
        room = "";
        previous.Clear();
    }

    public List<RoomInfo> Graph(Level level)
    {
        if (graph != null)
            return graph;
        var rooms = level.Session.MapData.Levels;
        graph = new List<RoomInfo>();
        foreach (LevelData data in rooms)
        {
            Rectangle b = data.Bounds;
            var a = new RoomRoute.Room(data.Name, b.Left, b.Top, b.Right, b.Bottom);
            string[] neighbors =
                rooms
                    .Where(other => other.Name != data.Name &&
                                    RoomRoute.Adjacent(
                                        a, new RoomRoute.Room(other.Name, other.Bounds.Left,
                                                              other.Bounds.Top, other.Bounds.Right,
                                                              other.Bounds.Bottom)))
                    .Select(other => other.Name)
                    .ToArray();
            bool terminal = data.Name == "end" || data.Name.StartsWith("end-");
            graph.Add(new RoomInfo(data.Name, neighbors, terminal));
        }
        return graph;
    }

    public List<Portal> Portals(Level level, Vector2 player)
    {
        var portals = new List<Portal>();
        Rectangle a = level.Bounds;
        foreach (LevelData other in level.Session.MapData.Levels)
        {
            if (other.Name == level.Session.Level)
                continue;
            Rectangle b = other.Bounds;
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
            Vector2? best = null;
            float distance = float.PositiveInfinity;
            foreach (Vector2 candidate in candidates)
            {
                Vector2 inside = new(Math.Clamp(candidate.X, a.Left + 8, a.Right - 8),
                                     Math.Clamp(candidate.Y, a.Top + 8, a.Bottom - 8));
                if (level.CollideCheck<Solid>(inside) || level.CollideCheck<Spikes>(inside))
                    continue;
                float d = Vector2.DistanceSquared(player, candidate);
                if (d < distance)
                {
                    best = candidate;
                    distance = d;
                }
            }
            if (best is Vector2 point)
                portals.Add(new Portal(other.Name, point.X, point.Y));
        }
        return portals;
    }

    public void NewChapter()
    {
        graph = null;
        Reset();
    }

    public RoomGrid Map(Level level)
    {
        if (grid != null && room == level.Session.Level)
            return grid;
        room = level.Session.Level;
        previous.Clear();
        Rectangle b = level.Bounds;
        int cell = Math.Max(8, (int)Math.Ceiling(Math.Max(b.Width, b.Height) / 128.0));
        int width = (int)Math.Ceiling(b.Width / (double)cell);
        int height = (int)Math.Ceiling(b.Height / (double)cell);
        var tiles = new char[width * height];
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
            {
                Vector2 point = new(b.X + (x + .5f) * cell, b.Y + (y + .5f) * cell);
                tiles[y * width + x] = level.CollideCheck<Spikes>(point)       ? 'X'
                                       : level.CollideCheck<SolidTiles>(point) ? '#'
                                                                               : '.';
            }
        grid = new RoomGrid(b.X, b.Y, cell, width, height, new string(tiles));
        Revision++;
        return grid;
    }

    public List<ObjectInfo> Objects(Level level, Vector2 player, int frame)
    {
        var found = new List<ObjectInfo>();
        var alive = new HashSet<Entity>();
        foreach (Entity entity in level.Entities)
        {
            if (!entity.Collidable || entity is Player)
                continue;
            string name = entity.GetType().Name;
            string kind = Kind(name);
            if (kind == "")
                continue;
            alive.Add(entity);
            Vector2 velocity = Vector2.Zero;
            if (previous.TryGetValue(entity, out var old) && frame > old.frame)
                velocity = (entity.Center - old.pos) * (60f / (frame - old.frame));
            previous[entity] = (entity.Center, frame);
            found.Add(new ObjectInfo(kind, name, entity.Center.X, entity.Center.Y, velocity.X,
                                     velocity.Y, entity.Width, entity.Height));
        }
        foreach (Entity gone in previous.Keys.Where(e => !alive.Contains(e)).ToArray())
            previous.Remove(gone);
        return found.OrderBy(e => Vector2.DistanceSquared(player, new Vector2(e.x, e.y)))
            .Take(16)
            .ToList();
    }

    private static string Kind(string name)
    {
        if (name.Contains("Spinner") || name.Contains("Seeker") || name.Contains("Lava") ||
            name.Contains("FireBall") || name.Contains("Oshiro") ||
            name.Contains("BadelineOldsite"))
            return "hazard";
        if (name.Contains("Refill") || name.Contains("Spring"))
            return "refill";
        if (name.Contains("Booster") || name.Contains("Feather") || name.Contains("FlyFeather"))
            return "booster";
        if (name.Contains("Key"))
            return "key";
        if (name.Contains("Switch"))
            return "switch";
        if (name.Contains("TheoCrystal") || name.Contains("Glider"))
            return "carry";
        if (name.Contains("Door") || name.Contains("TempleGate"))
            return "door";
        if (name.Contains("SummitCheckpoint") || name.Contains("BadelineBoost"))
            return "goal";
        if (name.Contains("Platform") || name.Contains("ZipMover") || name.Contains("MoveBlock") ||
            name.Contains("CrushBlock") || name.Contains("FallingBlock") ||
            name.Contains("DreamBlock"))
            return "platform";
        return "";
    }
}
