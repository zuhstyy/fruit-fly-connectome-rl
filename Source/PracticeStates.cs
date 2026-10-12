using System.Reflection;

namespace Celeste.Mod.FlyBrainCeleste;

// Keep the dependency in one small adapter. SpeedrunTool clones the whole level;
// FlyBrain never approximates a savestate by changing the player's position.
internal sealed class PracticeStates
{
    private Type? slots;
    private Type? exports;
    private readonly Dictionary<int, int> buttonStates = new();
    public bool Available { get; private set; }
    public string Error { get; private set; } = "SpeedrunTool is not loaded";

    public void Initialize(Type hudType)
    {
        foreach (Assembly assembly in AppDomain.CurrentDomain.GetAssemblies())
        {
            slots ??= assembly.GetType("Celeste.Mod.SpeedrunTool.SaveLoad.SaveSlotsManager");
            exports ??= assembly.GetType(
                "Celeste.Mod.SpeedrunTool.ModInterop.SaveLoadInterop+SaveLoadExports");
        }
        if (slots == null || exports == null)
            return;
        foreach (string method in new[] { "SaveStateTas", "LoadStateTas", "ClearStateTas" })
            if (slots.GetMethod(method, BindingFlags.Public | BindingFlags.Static, null,
                                new[] { typeof(string) }, null) == null)
            {
                Error = "SpeedrunTool practice API is incompatible: " + method;
                return;
            }
        // The HUD owns a module reference, which owns the TCP server. It must
        // never be cloned into the game state. This is an upstream interop API.
        GameAccess.Call(exports, "IgnoreSaveState", hudType, false);
        Available = true;
        Error = "";
    }

    public bool Capture(int slot, BotInputs inputs)
    {
        if (!Available)
            return false;
        bool ok = GameAccess.Call(slots!, "SaveStateTas", Name(slot)) is true;
        if (ok)
            buttonStates[slot] = inputs.Capture();
        else
            Error = "SpeedrunTool rejected capture (busy or unsafe state)";
        return ok;
    }

    public bool Restore(int slot, BotInputs inputs)
    {
        if (!Available || !buttonStates.ContainsKey(slot))
            return false;
        bool ok = GameAccess.Call(slots!, "LoadStateTas", Name(slot)) is true;
        if (ok)
            inputs.Restore(buttonStates[slot]);
        else
            Error = "SpeedrunTool rejected restore";
        return ok;
    }

    public void Clear()
    {
        if (Available)
            foreach (int slot in buttonStates.Keys)
                GameAccess.Call(slots!, "ClearStateTas", Name(slot));
        buttonStates.Clear();
    }

    public void ClearSlot(int slot)
    {
        if (Available && buttonStates.ContainsKey(slot))
            GameAccess.Call(slots!, "ClearStateTas", Name(slot));
        buttonStates.Remove(slot);
    }

    private static string Name(int slot) => "FlyBrain Practice " + slot;
}
