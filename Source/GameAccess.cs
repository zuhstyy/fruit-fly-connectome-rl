using System.Linq;
using System.Reflection;
using Monocle;

namespace Celeste.Mod.FlyBrainCeleste;

// These few version-sensitive APIs are checked at runtime, with visible failures.
internal static class GameAccess
{
    private const BindingFlags Flags =
        BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Static;
    public static object? Get(object obj, string name)
    {
        Type t = obj as Type ?? obj.GetType();
        object? target = obj is Type ? null : obj;
        return t.GetField(name, Flags)?.GetValue(target) ??
               t.GetProperty(name, Flags)?.GetValue(target);
    }
    public static bool Set(object obj, string name, object? value)
    {
        Type t = obj as Type ?? obj.GetType();
        object? target = obj is Type ? null : obj;
        if (t.GetField(name, Flags) is FieldInfo f && !f.IsInitOnly)
        {
            f.SetValue(target, value);
            return true;
        }
        if (t.GetProperty(name, Flags) is PropertyInfo p && p.CanWrite)
        {
            p.SetValue(target, value);
            return true;
        }
        return false;
    }
    public static object? Call(object obj, string name, params object?[] args)
    {
        Type t = obj as Type ?? obj.GetType();
        object? target = obj is Type ? null : obj;
        foreach (MethodInfo method in t.GetMethods(Flags).Where(m => m.Name == name))
        {
            ParameterInfo[] ps = method.GetParameters();
            if (ps.Length < args.Length || ps.Skip(args.Length).Any(p => !p.HasDefaultValue))
                continue;
            if (ps.Take(args.Length)
                    .Where((p, i) => args[i] != null && !p.ParameterType.IsInstanceOfType(args[i]))
                    .Any())
                continue;
            return method.Invoke(
                target, args.Concat(ps.Skip(args.Length).Select(p => p.DefaultValue)).ToArray());
        }
        throw new MissingMethodException(t.FullName, name);
    }
    public static string SavePath()
    {
        string root = Environment.GetEnvironmentVariable("EVEREST_SAVEPATH") ?? "";
        if (!Path.IsPathFullyQualified(root))
            throw new InvalidOperationException("Missing isolated save root.");
        string expected = Path.GetFullPath(Path.Combine(root, "Saves"));
        string actual = Path.GetFullPath((string)(Call(typeof(UserIO), "GetSaveFilePath", "0") ??
                                                  throw new InvalidOperationException(
                                                      "Cannot resolve the actual game save path")));
        string prefix =
            expected.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar) +
            Path.DirectorySeparatorChar;
        if (!actual.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
            throw new InvalidOperationException("Everest save path is not isolated: " + actual);
        return expected;
    }
    public static void ConfigureBoot()
    {
        // These are in-memory settings for a private worker, before GameLoader
        // starts. Do not require the user to press past the startup sequence.
        var core = typeof(EverestModule).Assembly.GetType("Celeste.Mod.Core.CoreModule");
        var settings = core == null ? null : Get(core, "Settings");
        if (settings != null)
        {
            Set(settings, "LaunchWithoutIntro", true);
            Set(settings, "AutoUpdateModsOnStartup", false);
        }
        Engine.Instance.InactiveSleepTime = TimeSpan.Zero;
        Set(Settings.Instance, "Fullscreen", false);
        Set(Settings.Instance, "WindowScale", 1);
        Set(Settings.Instance, "MusicVolume", 0);
        Set(Settings.Instance, "SFXVolume", 0);
        // Audio.Init has not run yet. ApplyVolumes/ApplyScreen belong to the
        // later chapter-start path, after the game finishes loading content.
    }
    public static void PrepareSave()
    {
        SavePath(); // Check the effective game path, not just the environment variable.
        if (SaveData.Instance != null)
            return;
        var data = new SaveData();
        Set(data, "Name", "Fly Brain Training");
        Call(typeof(SaveData), "Start", data, 0);
        if (SaveData.Instance == null)
            throw new InvalidOperationException(
                "SaveData.Start did not initialize a training save.");
    }
    public static void EnterChapter(Session session)
    {
        PrepareSave();
        SaveData.Instance.CurrentSession = session;
        SaveData.Instance.LastArea = session.Area;
        // Start with the normal LevelLoader/Player intro. No completion flags
        // or unlock cheats are applied to the isolated save.
        Engine.Scene = new LevelLoader(session, null);
    }
    public static void ConfigureBackground()
    {
        // Celeste's Monocle.Engine has no PauseOnFocusLost property. Requiring
        // it aborted every start request in 2.0. Worker pause requests made
        // while unfocused are handled by FlyBrainModule's Level.Pause hook.
        Engine.Instance.InactiveSleepTime = TimeSpan.Zero;
        var settings = Settings.Instance;
        Set(settings, "Fullscreen", false);
        Set(settings, "WindowScale", 1);
        // Settings application is optional: failure does not falsify readiness.
        try
        {
            Call(settings, "ApplyScreen");
        }
        catch (MissingMethodException)
        {
        }
        Set(settings, "MusicVolume", 0);
        Set(settings, "SFXVolume", 0);
        try
        {
            Call(settings, "ApplyVolumes");
        }
        catch (MissingMethodException)
        {
        }
    }
}
