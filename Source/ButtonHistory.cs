namespace Celeste.Mod.FlyBrainCeleste;

// One history for the worker, retained across TCP requests and decision holds.
// A dash press occurs only after release; holding dash does not auto-repeat it.
internal sealed class ButtonHistory
{
    public const int Jump = 1;
    public const int Dash = 2;
    public const int Grab = 4;
    public const int Interact = 8;

    private int current, previous;
    public int Capture() => previous;
    public void Restore(int saved)
    {
        current = previous = saved & 15;
    }
    public void Begin(Command action)
    {
        current = (action.Jump ? Jump : 0) | (action.Dash ? Dash : 0) | (action.Grab ? Grab : 0) |
                  (action.Interact ? Interact : 0);
    }
    public bool Check(int button) => (current & button) != 0;
    public bool Pressed(int button) => Check(button) && (previous & button) == 0;
    public bool Released(int button) => !Check(button) && (previous & button) != 0;
    public void End()
    {
        previous = current;
    }
    public void Reset()
    {
        current = previous = 0;
    }
}
