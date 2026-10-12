namespace Celeste.Mod.FlyBrainCeleste;

// Static map geometry establishes order independently of the bot's wandering.
// Graph distance from the chapter start is a heuristic, not a puzzle solver.
internal sealed class RoomRoute
{
    public readonly record struct Room(string Name, int Left, int Top, int Right, int Bottom);
    private readonly Dictionary<string, int> ranks = new();
    public int Rank(string name) => ranks.GetValueOrDefault(name, -1);
    public bool Forward(string from, string to) => Rank(from) >= 0 && Rank(to) > Rank(from);
    public static bool Adjacent(Room a, Room b) =>
        ((Math.Abs(b.Left - a.Right) <= 8 || Math.Abs(b.Right - a.Left) <= 8) &&
         Math.Min(a.Bottom, b.Bottom) - Math.Max(a.Top, b.Top) > 16) ||
        ((Math.Abs(b.Top - a.Bottom) <= 8 || Math.Abs(b.Bottom - a.Top) <= 8) &&
         Math.Min(a.Right, b.Right) - Math.Max(a.Left, b.Left) > 16);
    public RoomRoute(IReadOnlyList<Room> rooms, string start, bool prologue = false)
    {
        var byName = new Dictionary<string, Room>();
        foreach (Room r in rooms)
            byName[r.Name] = r;
        if (!byName.ContainsKey(start))
            return;
        // The vanilla Prologue starts in "0", next to the earlier car room "-1".
        // BFS from "0" incorrectly calls BOTH directions forward. The vanilla
        // Prologue's signed numeric room names encode its actual sequence.
        if (prologue)
        {
            var numbers = new Dictionary<string, int>();
            foreach (Room r in rooms)
                if (int.TryParse(r.Name, out int n))
                    numbers[r.Name] = n;
            if (numbers.ContainsKey("-1") && numbers.ContainsKey("0"))
            {
                int lowest = int.MaxValue;
                foreach (int n in numbers.Values)
                    lowest = Math.Min(lowest, n);
                foreach (var pair in numbers)
                    ranks[pair.Key] = pair.Value - lowest;
                return; // Non-numeric special rooms remain unknown, not guessed.
            }
        }
        var queue = new Queue<string>();
        ranks[start] = 0;
        queue.Enqueue(start);
        while (queue.Count > 0)
        {
            string name = queue.Dequeue();
            foreach (Room other in rooms)
            {
                if (ranks.ContainsKey(other.Name) || !Adjacent(byName[name], other))
                    continue;
                ranks[other.Name] = ranks[name] + 1;
                queue.Enqueue(other.Name);
            }
        }
    }
}
