using Microsoft.Xna.Framework;
using Monocle;

namespace Celeste.Mod.FlyBrainCeleste;

// Swap in virtual inputs only for a simulated gameplay frame, then restore them.
// Real keyboard/menu bindings are never changed or saved.
public sealed class BotInputs : IDisposable
{
    private Command action = new();
    private readonly ButtonHistory buttons = new();
    private readonly VirtualIntegerAxis x, y;
    private readonly VirtualJoystick aim;
    private readonly VirtualButton jump, dash, grab, interact;

    public BotInputs()
    {
        x = new VirtualIntegerAxis(new Axis(() => action.MoveX));
        y = new VirtualIntegerAxis(new Axis(() => action.MoveY));
        aim = new VirtualJoystick(true, new Stick(() => new Vector2(action.MoveX, action.MoveY)));
        jump = new VirtualButton(0f, new Button(ButtonHistory.Jump, buttons));
        dash = new VirtualButton(0f, new Button(ButtonHistory.Dash, buttons));
        grab = new VirtualButton(0f, new Button(ButtonHistory.Grab, buttons));
        interact = new VirtualButton(0f, new Button(ButtonHistory.Interact, buttons));
        // Update these exactly once ourselves, after the engine's MInput pass.
        Dispose();
    }

    public void Reset()
    {
        buttons.Reset();
        action = new Command();
    }

    public int Capture() => buttons.Capture();
    public void Restore(int saved) => buttons.Restore(saved);

    public void Run(Command next, Action update)
    {
        action = next;
        buttons.Begin(next);
        var oldX = Input.MoveX;
        var oldY = Input.MoveY;
        var oldAim = Input.Aim;
        var oldJump = Input.Jump;
        var oldDash = Input.Dash;
        var oldGrab = Input.Grab;
        var oldTalk = GameAccess.Get(typeof(Input), "Talk");
        var oldConfirm = GameAccess.Get(typeof(Input), "MenuConfirm");
        var oldFeather = GameAccess.Get(typeof(Input), "Feather");
        bool oldDisabled = MInput.Disabled;
        try
        {
            MInput.Disabled = false;
            x.Update();
            y.Update();
            aim.Update();
            jump.Update();
            dash.Update();
            grab.Update();
            interact.Update();
            Input.MoveX = x;
            Input.MoveY = y;
            Input.Aim = aim;
            Input.Jump = jump;
            Input.Dash = dash;
            Input.Grab = grab;
            GameAccess.Set(typeof(Input), "Talk", interact);
            GameAccess.Set(typeof(Input), "MenuConfirm", interact);
            GameAccess.Set(typeof(Input), "Feather", aim);
            Input.UpdateGrab();
            update();
        }
        finally
        {
            Input.MoveX = oldX;
            Input.MoveY = oldY;
            Input.Aim = oldAim;
            Input.Jump = oldJump;
            Input.Dash = oldDash;
            Input.Grab = oldGrab;
            GameAccess.Set(typeof(Input), "Talk", oldTalk);
            GameAccess.Set(typeof(Input), "MenuConfirm", oldConfirm);
            GameAccess.Set(typeof(Input), "Feather", oldFeather);
            Input.ResetGrab();
            MInput.Disabled = oldDisabled;
            buttons.End();
            action = new Command();
        }
    }

    public void Dispose()
    {
        x.Deregister();
        y.Deregister();
        aim.Deregister();
        jump.Deregister();
        dash.Deregister();
        grab.Deregister();
        interact.Deregister();
    }

    private sealed class Axis(Func<float> value) : VirtualAxis.Node
    {
        public override float Value => value();
    }
    private sealed class Stick(Func<Vector2> value) : VirtualJoystick.Node
    {
        public override Vector2 Value => value();
    }
    private sealed class Button(int button, ButtonHistory history) : VirtualButton.Node
    {
        public override bool Check => history.Check(button);
        public override bool Pressed => history.Pressed(button);
        public override bool Released => history.Released(button);
    }
}
