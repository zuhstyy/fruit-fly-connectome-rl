using System.Diagnostics;
using Microsoft.Xna.Framework;

namespace Celeste.Mod.FlyBrainCeleste;

// The outer FNA loop can run freely; each physics update still receives 1/60 s.
// SuppressDraw is essential: skipping Engine.Draw alone still presents/vsyncs.
internal sealed class SimulationPacing
{
    private bool training, previousFixed;
    private long lastDraw;
    public bool Fast => training;

    public void SetTraining(Game game, bool enabled)
    {
        if (training == enabled)
            return;
        if (enabled)
        {
            previousFixed = game.IsFixedTimeStep;
            game.IsFixedTimeStep = false;
        }
        else
        {
            game.IsFixedTimeStep = previousFixed;
            game.ResetElapsedTime();
        }
        training = enabled;
    }

    public void PrepareDraw(Game game, bool visible)
    {
        if (!training)
            return;
        if (!visible || Stopwatch.GetElapsedTime(lastDraw).TotalSeconds < 1.0 / 30)
            game.SuppressDraw();
        else
            lastDraw = Stopwatch.GetTimestamp();
    }
}
