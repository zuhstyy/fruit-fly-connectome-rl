using Celeste.Mod.FlyBrainCeleste;
var history = new ButtonHistory();
int checks = 0;
void Check(bool valid, string reason)
{
    if (!valid)
        throw new Exception(reason);
    checks++;
}
int presses = 0, releases = 0, held = 0;
// Four distinct decisions, same buttons: 120 continuous frames, not four taps.
for (int decision = 0; decision < 4; decision++)
{
    var action = new Command { MoveX = 1,   Jump = true,     Grab = true,
                               Dash = true, Interact = true, Frames = 30 };
    for (int frame = 0; frame < action.Frames; frame++)
    {
        history.Begin(action);
        if (history.Pressed(1))
            presses++;
        if (history.Released(1))
            releases++;
        if (history.Check(1))
            held++;
        if (decision > 0 || frame > 0)
            Check(!history.Pressed(2), "Dash must not re-press during hold");
        Check(history.Check(4) && history.Check(8), "Grab/interact remain held");
        Check(action.MoveX == 1, "Jump must not change horizontal direction");
        history.End();
    }
}
Check(presses == 1 && releases == 0 && held == 120, "Continuous 120-frame hold");
history.Begin(new Command());
Check(history.Released(1) && history.Released(2) && history.Released(4) && history.Released(8),
      "All release edges");
history.End();
history.Begin(new Command { Dash = true });
Check(history.Pressed(2), "A new dash requires release then press");
history.End();
history.Reset();
history.Begin(new Command { Jump = true });
Check(history.Pressed(1), "Respawn clears prior hold");
Console.WriteLine($"{checks} button timing assertions passed (120 frames across 4 decisions).");
